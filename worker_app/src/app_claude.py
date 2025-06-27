import os
import uuid
import asyncio
from asyncio import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from tempfile import TemporaryDirectory

from auth import user_auth_dependency

app = FastAPI(
    description="Monster Neo Code Runtime Worker",
    docs_url="/docs",
    redoc_url="/"
)

# Global management for sessions and jobs
sessions = {}  # coding_session_id -> list of directories
temp_dirs = {}  # job_id -> PersistentTemporaryDirectory object
MAX_SESSIONS = 5  # Limit to 5 sessions per user

class JobStatus(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the job")
    status: str = Field(..., description="Current status of the job")
    stdout: Optional[str] = Field(None, description="Standard output of the subprocess")
    stderr: Optional[str] = Field(None, description="Standard error of the subprocess")
    exit_code: Optional[int] = Field(None, description="Exit code of the subprocess")

class CommandRequest(BaseModel):
    coding_session_id: str = Field(..., description="Session identifier to run commands within")
    command: str = Field(..., description="Command to be executed")
    detach: bool = Field(False, description="Whether to detach the process or wait for completion")
    workdir: Optional[str] = Field(None, description="Working directory for the command")

class CodeRequest(BaseModel):
    coding_session_id: str = Field(..., description="Session identifier for code execution")
    filename: str = Field(..., description="Filename to save the code")
    code: str = Field(..., description="Source code to be written to the file")
    workdir: Optional[str] = Field(None, description="Working directory for the file")

class SessionRequest(BaseModel):
    coding_session_id: str = Field(..., description="Identifier for the session to be manipulated")

class JobInfo(BaseModel):
    process: Any
    logs: Dict[str, str] = Field(default_factory=lambda: {"stdout": "", "stderr": ""})

jobs: Dict[str, JobInfo] = {}

class PersistentTemporaryDirectory:
    def __init__(self, coding_session_id):
        self._temp_dir = TemporaryDirectory()
        self.coding_session_id = coding_session_id

    @property
    def name(self):
        return self._temp_dir.name

    def cleanup(self):
        self._temp_dir.cleanup()

async def read_stream(stream: asyncio.streams.StreamReader, job_id: str, stream_name: str):
    while True:
        line = await stream.readline()
        if not line:
            break
        line_str = line.decode().strip()
        jobs[job_id].logs[stream_name] += line_str + "\n"

async def run_command(command: str, work_dir: str, job_id: str):
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=work_dir
    )
    
    jobs[job_id] = JobInfo(process=process)
    
    stdout_task = asyncio.create_task(read_stream(process.stdout, job_id, "stdout"))
    stderr_task = asyncio.create_task(read_stream(process.stderr, job_id, "stderr"))
    
    await process.wait()
    await stdout_task
    await stderr_task

@app.post("/session/create", response_model=dict, status_code=201)
def create_session(_ = Depends(user_auth_dependency)):
    coding_session_id = str(uuid.uuid4())
    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(status_code=403, detail="Maximum session limit reached")
    
    temp_dir = PersistentTemporaryDirectory(coding_session_id)
    sessions[coding_session_id] = []
    temp_dirs[coding_session_id] = temp_dir

    return {"coding_session_id": coding_session_id, "temp_dir": temp_dir.name}

@app.delete("/session/close/{coding_session_id}", response_model=dict, status_code=200)
def close_session(coding_session_id: str, _ = Depends(user_auth_dependency)):
    if coding_session_id in sessions:
        for workdir in sessions[coding_session_id]:
            if workdir in temp_dirs:
                temp_dirs[workdir].cleanup()
                del temp_dirs[workdir]
        del sessions[coding_session_id]
        return {"coding_session_id": coding_session_id, "status": "closed"}
    else:
        raise HTTPException(status_code=404, detail="Session ID not found")

@app.post("/subprocess/write_code", response_model=dict, status_code=200)
async def write_code(request: CodeRequest, _ = Depends(user_auth_dependency)):
    if request.coding_session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID! Please create session first!")
    
    if request.workdir is not None:
        if request.workdir not in sessions.get(request.coding_session_id, []):
            raise HTTPException(status_code=400, detail="Invalid or missing workdir for this session!")
        work_dir = request.workdir
    else:
        work_dir = temp_dirs[request.coding_session_id].name

    try:
        file_path = Path(work_dir) / request.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(request.code)
        return {"file_path": str(file_path), "workdir": request.workdir, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write code to file: {str(e)}")

@app.post("/subprocess/run", response_model=dict, status_code=200)
async def run_subprocess(request: CommandRequest, _ = Depends(user_auth_dependency)):
    if request.coding_session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID! Please create session first!")

    work_dir = temp_dirs[request.coding_session_id].name if request.workdir is None else request.workdir

    if "python" in request.command:
        request.command = f"python -u {request.command.split('python', 1)[1].strip()}"

    if request.detach:
        job_id = str(uuid.uuid4())
        asyncio.create_task(run_command(request.command, work_dir, job_id))
        return {"job_id": job_id, "status": "running"}
    else:
        process = await asyncio.create_subprocess_shell(
            request.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir
        )
        stdout, stderr = await process.communicate()
        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "exit_code": process.returncode,
            "workdir": request.workdir
        }

@app.get("/subprocess/logs/{job_id}", response_model=dict, status_code=200)
async def get_job_logs(job_id: str, _ = Depends(user_auth_dependency)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found or logs unavailable")
    
    job_info = jobs[job_id]
    process = job_info.process
    status = "running" if process.returncode is None else "completed"

    return {
        "stdout": job_info.logs["stdout"],
        "stderr": job_info.logs["stderr"],
        "status": status
    }

@app.delete("/subprocess/terminate/{job_id}", response_model=dict, status_code=200)
async def terminate_subprocess(job_id: str, _ = Depends(user_auth_dependency)):
    if job_id in jobs:
        process = jobs[job_id].process
        process.terminate()  # Sends SIGTERM
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()  # Force kill if it doesn't terminate in 5 seconds

        del jobs[job_id]
        return {"job_id": job_id, "status": "terminated"}
    else:
        raise HTTPException(status_code=404, detail="Job ID not found")

@app.get("/subprocess/status/{job_id}", response_model=JobStatus, status_code=200)
async def get_job_status(job_id: str, _ = Depends(user_auth_dependency)):
    if job_id in jobs:
        job_info = jobs[job_id]
        process = job_info.process
        if process.returncode is not None:  # Process has completed
            stdout = job_info.logs["stdout"]
            stderr = job_info.logs["stderr"]

            del jobs[job_id]

            return JobStatus(
                job_id=job_id,
                status='completed',
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode
            )
        else:
            return JobStatus(
                job_id=job_id,
                status='running',
            )
    else:
        raise HTTPException(status_code=404, detail="Job ID not found")

@app.get("/session/{coding_session_id}/files/{file_path:path}", response_model=None, status_code=200)
async def get_file(coding_session_id: str, file_path: str, _ = Depends(user_auth_dependency)):
    if coding_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    work_dir = temp_dirs.get(coding_session_id)
    if not work_dir:
        raise HTTPException(status_code=404, detail="Working directory not found for session")

    full_path = Path(work_dir.name) / file_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=full_path, filename=file_path.split("/")[-1])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)