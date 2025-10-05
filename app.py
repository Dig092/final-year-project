from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from typing import List, Literal
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
import importlib
import asyncio
import codecs
import sys
import os

app = FastAPI()

# Global variables
thread_managers = {}

# Attempt to import AutogenBackendThreadManager
try:
    from RuntimeManager import AutogenBackendThreadManager
except ImportError:
    print("Warning: Unable to import AutogenBackendThreadManager. Make sure RuntimeManager.py is in the correct location.")
    AutogenBackendThreadManager = None

# Pydantic model for initializing chat request body
class InitChatRequest(BaseModel):
    threadId: str
    message: str
    mode: Literal["cpu", "gpu"]
    container_id: str

# Pydantic model for sending user input
class UserInputRequest(BaseModel):
    input: str

# Function to reset global variables
def reset_globals():
    global thread_managers
    thread_managers = {}
    # Add any other global variables that need to be reset

# Function to reload modules
def reload_modules():
    global AutogenBackendThreadManager
    try:
        if 'RuntimeManager' in sys.modules:
            importlib.reload(sys.modules['RuntimeManager'])
        else:
            import RuntimeManager
        from RuntimeManager import AutogenBackendThreadManager
    except ImportError:
        print("Warning: Unable to reload AutogenBackendThreadManager. Make sure RuntimeManager.py is in the correct location.")
        AutogenBackendThreadManager = None

async def run_init_chat_in_background(thread_id, manager, message):
    try:
        # Run the async init_chat function
        await manager.a_init_chat(message)
    finally:
        # Ensure cleanup by removing the manager from thread_managers after completion
        thread_managers.pop(thread_id, None)


# Function to reinitialize the FastAPI app
def reinitialize_app():
    global app
    app = FastAPI()
    
    # Re-add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Re-register all routes
    app.get("/")(root)
    app.post("/init-chat")(init_chat)
    app.get("/events/{threadId}")(get_events)
    app.post("/send-user-input/{threadId}")(send_user_input)
    app.get("/user-input-required/{threadId}")(user_input_required)
    app.get("/get-artifacts")(get_artifacts)
    app.get("/agent_terminal_logs")(get_last_lines)
    app.delete("/terminate/{threadId}")(terminate_thread)
    app.delete("/reset")(reset_all)

# Root route for Hello World message
@app.get("/")
async def root():
    return {"message": "Hello, World!"}

# Initialize a new chat for a given threadId
@app.post("/init-chat")
async def init_chat(request: InitChatRequest, background_tasks: BackgroundTasks):
    # Check if AutogenBackendThreadManager is available
    if AutogenBackendThreadManager is None:
        raise HTTPException(status_code=500, detail="AutogenBackendThreadManager is not available")
    thread_id = request.threadId
    message = request.message
    # Validate request parameters
    if not thread_id or not message:
        raise HTTPException(status_code=400, detail="Missing 'threadId' or 'message' in request")
    # Check if the thread is already running
    if thread_id in thread_managers:
        return {"status": "Thread already running", "threadId": thread_id}
    # Initialize a new manager and store it in the thread_managers dictionary
    manager = AutogenBackendThreadManager(mode=request.mode, continer_id=request.container_id, thread_id=thread_id)
    manager.groupchat.messages.append({"content":message,"role":"user","name":"user"})
    thread_managers[thread_id] = manager
    # Run a_init_chat in the background
    background_tasks.add_task(run_init_chat_in_background, thread_id, manager, message)
    return {"status": "Chat initialized successfully", "threadId": thread_id}

# Get events for a specific thread
@app.get("/events/{threadId}")
async def get_events(threadId: str):
    manager = thread_managers.get(threadId)
    
    if not manager:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Get events asynchronously
    events = manager.get_events()

    # clenup events
    # remove all items in list where content == None or content == ""
    try: 
        events = [event for event in events if event["content"] != 'None' or event["content"] != ""]
    except Exception as e:
        pass
    
    # print(events)
    return events

# Send user input to a specific thread
@app.post("/send-user-input/{threadId}")
async def send_user_input(threadId: str, request: UserInputRequest):
    manager = thread_managers.get(threadId)

    if not manager:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Handle user input
    user_input = request.input
    manager.given_user_input.append(user_input)

    return {"status": "User input received", "input": user_input}

# Check if user input is required for a specific thread
@app.get("/user-input-required/{threadId}")
async def user_input_required(threadId: str):
    manager = thread_managers.get(threadId)

    if not manager:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Check the user_input_required flag
    input_required = manager.user_input_required
    return {"threadId": threadId, "user_input_required": input_required}

IGNORED_FILES = {
    "MonsterRuntimeAgent", "OAI_CONFIG_LIST_EXAMPLE", "RuntimeManager.py", "app.py",
    "monster_dockerapi", "scripts", "tmp", "worker_app", "OAI_CONFIG_LIST",
    "Readme.MD", "__pycache__", "description_obfuscated.md", "run.py",
    "setup.py", "venv"
}

def convert_file_to_bytes(file_path):
    """
    Convert a file to bytes if it is not in the ignored list.

    :param file_path: Full file path
    :return: Tuple of (file_name, file_bytes) or None if the file is ignored or not valid
    """
    file_name = os.path.basename(file_path)

    if file_name in IGNORED_FILES or not os.path.isfile(file_path):
        return None

    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        return file_name, file_bytes
    except Exception as e:
        print(f"Error reading file '{file_name}': {e}")
        return None

@app.get("/get-artifacts")
async def get_artifacts():
    """
    Converts all non-ignored files in the directory to bytes and returns them as key-value pairs.
    """
    current_dir = os.getcwd()
    artifacts = {}

    # Iterate through files in the directory
    for file_name in os.listdir(current_dir):
        file_path = os.path.join(current_dir, file_name)
        file_data = convert_file_to_bytes(file_path)
        if file_data:
            artifacts[file_data[0]] = file_data[1]  # file_name: file_bytes
        os.remove(file_path)
    if not artifacts:
        raise HTTPException(status_code=404, detail="No artifacts found")

    return artifacts

def read_last_n_lines_Old(filename: str, n_lines: int) -> List[str]:
    """
    Read the last n_lines from a file efficiently.
    """
    try:
        with open(filename, 'rb') as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            buffer = bytearray()
            lines = []
            pointer_location = file_size
            while pointer_location >= 0 and len(lines) < n_lines:
                f.seek(pointer_location)
                read_byte = f.read(1)
                if read_byte == b'\n' and buffer:
                    lines.append(buffer.decode()[::-1])
                    buffer = bytearray()
                else:
                    buffer.extend(read_byte)
                pointer_location -= 1
            if buffer:
                lines.append(buffer.decode()[::-1])
            return lines[::-1]  # Reverse to get correct order
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def read_last_n_lines(filename: str, n_lines: int) -> List[str]:
    """
    Efficiently read the last n_lines from a file, optimized for large files.
    """
    try:
        with open(filename, 'rb') as f:
            f.seek(0, os.SEEK_END)
            file_size = remaining_size = f.tell()
            chunk_size = 4096
            lines = []

            while remaining_size > 0 and len(lines) < n_lines:
                chunk_size = min(chunk_size, remaining_size)
                remaining_size -= chunk_size
                f.seek(remaining_size)
                chunk = f.read(chunk_size)
                
                lines_in_chunk = chunk.split(b'\n')

                if len(lines) > 0:
                    lines_in_chunk[-1] = lines_in_chunk[-1] + lines[0]
                    lines = lines_in_chunk[-1:] + lines[1:]
                else:
                    lines = lines_in_chunk

                lines = lines[-n_lines:]  # Keep only the last n_lines

            return [decode_line(line) for line in reversed(lines) if line]

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

def read_last_n_lines_old1(filename: str, n_lines: int) -> List[str]:
    """
    Read the last n_lines from a file efficiently, handling potential encoding issues.
    """
    try:
        with open(filename, 'rb') as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            buffer = bytearray()
            lines = []
            pointer_location = file_size
            while pointer_location >= 0 and len(lines) < n_lines:
                f.seek(pointer_location)
                read_byte = f.read(1)
                if read_byte == b'\n' and buffer:
                    lines.append(decode_line(buffer[::-1]))
                    buffer = bytearray()
                else:
                    buffer.extend(read_byte)
                pointer_location -= 1
            if buffer:
                lines.append(decode_line(buffer[::-1]))
            return lines[::-1]  # Reverse to get correct order
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def decode_line(line: bytes) -> str:
    """
    Attempt to decode a line, falling back to different methods if UTF-8 fails.
    """
    encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
    for encoding in encodings:
        try:
            return line.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    # If all else fails, use UTF-8 with error replacement
    return codecs.decode(line, 'utf-8', errors='replace')


@app.get("/agent_terminal_logs")
def get_last_lines(n_lines: int = Query(10, ge=1)):
    """
    FastAPI route to get the last n_lines from a file.
    """
    filename = "/tmp/streamneo.log"  # Change this to your specific file path
    lines = read_last_n_lines(filename, n_lines)
    return {"last_lines": lines}

@app.delete("/terminate/{threadId}")
async def terminate_thread(threadId: str, request: Request):
    manager = thread_managers.pop(threadId, None)

    if not manager:
        raise HTTPException(status_code=404, detail="Thread not found")

    try:
        manager.monster_executor.cleanup()
    except Exception as e:
        pass
    
    # Clean up resources
    try:
        if hasattr(manager, 'terminate_thread'):
            manager.terminate_thread()
        elif hasattr(manager, 'cleanup'):
            await manager.cleanup()
        else:
            print(f"Warning: No termination or cleanup method found for thread {threadId}")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

    # Reset globals and reload app
#    reset_globals()
#    reload_modules()
#    reinitialize_app()

    # Optionally, you can add a small delay to ensure everything is reset
#    await asyncio.sleep(1)

    return {"status": "Thread terminated and app reloaded", "threadId": threadId}

@app.delete("/reset")
async def reset_all():
    global thread_managers
    
    for thread_id, manager in list(thread_managers.items()):
        try:
            if hasattr(manager, 'monster_executor'):
                manager.monster_executor.cleanup()
        except Exception as e:
            print(f"Error cleaning up monster_executor for thread {thread_id}: {str(e)}")
        
        try:
            if hasattr(manager, 'terminate_thread'):
                manager.terminate_thread()
            elif hasattr(manager, 'cleanup'):
                await manager.cleanup()
            else:
                print(f"Warning: No termination or cleanup method found for thread {thread_id}")
        except Exception as e:
            print(f"Error during cleanup for thread {thread_id}: {str(e)}")
    
    # Clear all thread managers
    thread_managers.clear()
    
    # Reset globals and reload app
    reset_globals()
    reload_modules()
    reinitialize_app()
    
    # Add a small delay to ensure everything is reset
    await asyncio.sleep(1)
    
    return {"status": "All threads terminated and app reloaded"}

# Make sure to call reinitialize_app() at the end of your script
reinitialize_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, loop="asyncio")
