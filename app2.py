from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks, status
from pydantic import BaseModel
from typing import List, Literal, Dict, TYPE_CHECKING
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from asyncio import TimeoutError
import importlib
import signal
import psutil
import codecs
import sys
import os
import threading
import multiprocessing
import queue

# Configuration
EVENT_TIMEOUT = 30  # seconds
CLEANUP_TIMEOUT = 10  # seconds
IGNORED_FILES = {
    "MonsterRuntimeAgent", "OAI_CONFIG_LIST_EXAMPLE", "RuntimeManager.py", "app.py",
    "monster_dockerapi", "scripts", "tmp", "worker_app", "OAI_CONFIG_LIST",
    "Readme.MD", "__pycache__", "description_obfuscated.md", "run.py",
    "setup.py", "venv"
}

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attempt to import AutogenBackendThreadManager
try:
    from RuntimeManager import AutogenBackendThreadManager
except ImportError:
    print("Warning: Unable to import AutogenBackendThreadManager. Make sure RuntimeManager.py is in the correct location.")
    AutogenBackendThreadManager = None

if TYPE_CHECKING:
    from .thread_process import ThreadProcess

async def run_init_chat_in_background(thread_id: str, thread_process: "ThreadProcess", message: str):
    try:
        # Run the async init_chat function
        await thread_process.thread_manager.a_init_chat(message)
    except Exception as e:
        print(f"Error in background task for thread {thread_id}: {str(e)}")
    finally:
        # Ensure cleanup by removing the thread from thread manager
        await thread_process.terminate()
        thread_manager.threads.pop(thread_id, None)

class AsyncEventManager:
    def __init__(self):
        self._queue = asyncio.Queue()
        self.is_running = True

    async def add_event(self, event):
        if self.is_running:
            await self._queue.put(event)

    async def get_events(self):
        events = []
        while not self._queue.empty() and self.is_running:
            try:
                event = self._queue.get_nowait()
                events.append(event)
            except asyncio.QueueEmpty:
                break
        return events

    async def cleanup(self):
        self.is_running = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

class ThreadProcess:
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.process = None
        self.event_manager = AsyncEventManager()
        self.process_ids = set()
        self.stop_event = threading.Event()
        self.thread_manager = None
        self.is_running = False

    async def run_manager(self, mode: str, container_id: str, message: str):
        self.is_running = True
        try:
            self.thread_manager = AutogenBackendThreadManager(mode=mode, continer_id=container_id, thread_id=self.thread_id)
            self.thread_manager.groupchat.messages.append({"content": message, "role": "user", "name": "user"})
            
            # Register the main process
            self.add_process_id(os.getpid())
            
            # Start the thread manager and capture events
            while not self.stop_event.is_set():
                events = self.thread_manager.get_events()
                if events:
                    for event in events:
                        await self.event_manager.add_event(event)
                await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse
                
            # If monster_executor exists, track its processes
            if hasattr(self.thread_manager, 'monster_executor'):
                if hasattr(self.thread_manager.monster_executor, 'process_ids'):
                    for pid in self.thread_manager.monster_executor.process_ids:
                        self.add_process_id(pid)

        except Exception as e:
            print(f"Error in thread {self.thread_id}: {str(e)}")
        finally:
            self.is_running = False

    async def terminate(self):
        """Terminate this specific thread and its processes"""
        self.stop_event.set()
        self.is_running = False
        
        try:
            # Stop the event manager
            await self.event_manager.cleanup()

            # Clean up thread manager
            if self.thread_manager:
                if hasattr(self.thread_manager, 'monster_executor'):
                    try:
                        self.thread_manager.monster_executor.cleanup()
                    except Exception as e:
                        print(f"Error cleaning monster executor: {str(e)}")
                
                if hasattr(self.thread_manager, 'terminate_thread'):
                    try:
                        self.thread_manager.terminate_thread()
                    except Exception as e:
                        print(f"Error in terminate_thread: {str(e)}")
                elif hasattr(self.thread_manager, 'cleanup'):
                    try:
                        await self.thread_manager.cleanup()
                    except Exception as e:
                        print(f"Error in cleanup: {str(e)}")

            # Get all processes from monster_executor
            if hasattr(self.thread_manager, 'monster_executor'):
                if hasattr(self.thread_manager.monster_executor, 'process_ids'):
                    self.process_ids.update(self.thread_manager.monster_executor.process_ids)

            # Terminate all registered processes for this thread
            for pid in list(self.process_ids):
                if pid != os.getpid():  # Don't kill the main API process
                    try:
                        process = psutil.Process(pid)
                        children = process.children(recursive=True)
                        
                        # Terminate child processes first
                        for child in children:
                            try:
                                child.terminate()
                            except psutil.NoSuchProcess:
                                pass
                        
                        # Wait for children to terminate
                        _, alive = psutil.wait_procs(children, timeout=3)
                        
                        # Kill any remaining children
                        for child in alive:
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                        
                        # Terminate the process
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            process.kill()
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        print(f"Error terminating process {pid}: {str(e)}")
                    finally:
                        self.remove_process_id(pid)

        except Exception as e:
            print(f"Error terminating thread {self.thread_id}: {str(e)}")

# Global thread manager
class ThreadManager:
    def __init__(self):
        self.threads: Dict[str, ThreadProcess] = {}

    async def create_thread(self, thread_id: str, mode: str, container_id: str, message: str):
        if thread_id in self.threads:
            return False
        
        thread_process = ThreadProcess(thread_id)
        self.threads[thread_id] = thread_process
        
        # Start the thread process asynchronously
        asyncio.create_task(thread_process.run_manager(mode, container_id, message))
        return True

    async def terminate_thread(self, thread_id: str):
        thread_process = self.threads.get(thread_id)
        if thread_process:
            await thread_process.terminate()
            self.threads.pop(thread_id, None)
            return True
        return False

    async def terminate_all(self):
        tasks = []
        for thread_id in list(self.threads.keys()):
            tasks.append(self.terminate_thread(thread_id))
        if tasks:
            await asyncio.gather(*tasks)

# Initialize global thread manager
thread_manager = ThreadManager()

# Pydantic models
class InitChatRequest(BaseModel):
    threadId: str
    message: str
    mode: Literal["cpu", "gpu"]
    container_id: str

class UserInputRequest(BaseModel):
    input: str

# Routes
@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/init-chat")
async def init_chat(request: InitChatRequest, background_tasks: BackgroundTasks):
    if AutogenBackendThreadManager is None:
        raise HTTPException(status_code=500, detail="AutogenBackendThreadManager is not available")
    
    if not request.threadId or not request.message:
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    # Check if thread already exists
    if request.threadId in thread_manager.threads:
        return {"status": "Thread already running", "threadId": request.threadId}
    
    # Create new thread process
    thread_process = ThreadProcess(request.threadId)
    thread_process.thread_manager = AutogenBackendThreadManager(
        mode=request.mode, 
        continer_id=request.container_id, 
        thread_id=request.threadId
    )
    thread_process.thread_manager.groupchat.messages.append({
        "content": request.message, 
        "role": "user", 
        "name": "user"
    })
    
    # Store in thread manager
    thread_manager.threads[request.threadId] = thread_process
    
    # Start background task
    background_tasks.add_task(
        run_init_chat_in_background, 
        request.threadId, 
        thread_process, 
        request.message
    )
    
    return {"status": "Chat initialized successfully", "threadId": request.threadId}

@app.get("/events/{threadId}")
async def get_events(threadId: str):
    thread_process = thread_manager.threads.get(threadId)
    if not thread_process:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    events = await thread_process.event_manager.get_events()
    return [event for event in events if event.get("content")]

@app.post("/send-user-input/{threadId}")
async def send_user_input(threadId: str, request: UserInputRequest):
    thread_process = thread_manager.threads.get(threadId)
    if not thread_process:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    if thread_process.thread_manager:
        thread_process.thread_manager.given_user_input.append(request.input)
        return {"status": "User input received", "input": request.input}
    
    raise HTTPException(status_code=400, detail="Thread manager not initialized")

@app.get("/user-input-required/{threadId}")
async def user_input_required(threadId: str):
    thread_process = thread_manager.threads.get(threadId)
    if not thread_process:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    if thread_process.thread_manager:
        return {
            "threadId": threadId,
            "user_input_required": thread_process.thread_manager.user_input_required
        }
    
    raise HTTPException(status_code=400, detail="Thread manager not initialized")

@app.delete("/terminate/{threadId}")
async def terminate_thread(threadId: str):
    thread_process = thread_manager.threads.get(threadId)
    if not thread_process:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    await thread_process.terminate()
    thread_manager.threads.pop(threadId, None)
    
    return {"status": "Thread terminated", "threadId": threadId}

@app.delete("/reset")
async def reset_all():
    await thread_manager.terminate_all()
    return {"status": "All threads terminated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, loop="asyncio")