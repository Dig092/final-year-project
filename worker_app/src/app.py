import os
import uuid
import select 

from pathlib import Path
from typing import Optional
from subprocess import Popen, PIPE
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, Depends
from tempfile import TemporaryDirectory, NamedTemporaryFile

from auth import user_auth_dependency

app = FastAPI(
    description = "Monster Neo Code Runtime Worker",
    docs_url = "/docs",
    redoc_url = "/"
)

# Global management for sessions and jobs
sessions = {}  # coding_session_id -> list of directories
jobs = {}  # job_id -> Popen object and metadata
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


class PersistentTemporaryDirectory:
    def __init__(self, coding_session_id):
        self._temp_dir = TemporaryDirectory()
        self.coding_session_id = coding_session_id

    @property
    def name(self):
        return self._temp_dir.name

    def cleanup(self):
        self._temp_dir.cleanup()

@app.post("/session/create", response_model=dict, status_code=201, responses={
    201: {"description": "Session successfully created with a temporary directory"},
    403: {"description": "Maximum session limit reached"},
    401: {"description": "Unauthorized - authentication required"}
})
def create_session(_ = Depends(user_auth_dependency)):
    """
    Create a new coding session with a unique session ID and temporary directory.
    
    - Returns 201 if the session is created successfully.
    - Returns 403 if the maximum session limit is reached.
    - Returns 401 if unauthorized.
    """
    coding_session_id = str(uuid.uuid4())
    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(status_code=403, detail="Maximum session limit reached")
    
    # Create and track a new persistent temporary directory for this session
    temp_dir = PersistentTemporaryDirectory(coding_session_id)
    sessions[coding_session_id] = []
    temp_dirs[coding_session_id] = temp_dir  # Link the temp directory with the session ID

    return {"coding_session_id": coding_session_id, "temp_dir": temp_dir.name}

@app.delete("/session/close/{coding_session_id}", response_model=dict, status_code=200, responses={
    200: {"description": "Session successfully closed"},
    404: {"description": "Session ID not found"},
    401: {"description": "Unauthorized - authentication required"}
})
def close_session(coding_session_id: str, _ = Depends(user_auth_dependency)):
    """
    Close the specified coding session and clean up any associated resources.
    
    - Returns 200 if the session is successfully closed.
    - Returns 404 if the session ID is not found.
    - Returns 401 if unauthorized.
    """
    if coding_session_id in sessions:
        for workdir in sessions[coding_session_id]:
            if workdir in temp_dirs:
                temp_dirs[workdir].cleanup()
                del temp_dirs[workdir]
        del sessions[coding_session_id]
        return {"coding_session_id": coding_session_id, "status": "closed"}
    else:
        raise HTTPException(status_code=404, detail="Session ID not found")

@app.post("/subprocess/write_code", response_model=dict, status_code=200, responses={
    200: {"description": "Code successfully written to the file"},
    400: {"description": "Invalid session ID or missing workdir"},
    500: {"description": "Failed to write code to file due to server error"},
    401: {"description": "Unauthorized - authentication required"}
})
async def write_code(request: CodeRequest, _ = Depends(user_auth_dependency)):
    if request.coding_session_id not in sessions:
        raise HTTPException(status_code=400, detail = "invalid session id! Please create session first!")
    print(request.workdir)
    if request.workdir != None:
        if request.workdir not in sessions.get(request.coding_session_id, []):
            raise HTTPException(status_code=400, detail="Invalid or missing workdir for this session!")
        else:
            work_dir = request.workdir
    else:
        work_dir = temp_dirs[request.coding_session_id].name

    print("Work Dir", work_dir)
    
    try:
        file_path = Path(work_dir) / request.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(request.code)
        return {"file_path": str(file_path), "workdir": request.workdir, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write code to file: {str(e)}")

@app.post("/subprocess/run", response_model=dict, status_code=200, responses={
    200: {"description": "Command successfully executed"},
    400: {"description": "Invalid session ID or missing workdir"},
    401: {"description": "Unauthorized - authentication required"}
})
async def run_subprocess(request: CommandRequest, _ = Depends(user_auth_dependency)):
    """
    Run a shell command in the session's temporary directory.
    
    - Returns 200 with the command's output and exit code (synchronous).
    - If detach=True, returns a job ID for later tracking.
    - Returns 400 for invalid session ID or workdir.
    - Returns 401 if unauthorized.
    """
    if request.coding_session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID! Please create session first!")

    # Determine the working directory
    if request.workdir is not None:
        if request.workdir not in sessions.get(request.coding_session_id, []):
            raise HTTPException(status_code=400, detail="Invalid or missing workdir for this session!")
        work_dir = request.workdir
    else:
        work_dir = temp_dirs[request.coding_session_id].name

    if "python" in request.command:
        request.command = f"python -u {request.command.split('python', 1)[1].strip()}"

    # Handle detached process (run asynchronously)
    if request.detach:
        job_id = str(uuid.uuid4())  # Generate a unique job ID

        # Redirect stdout and stderr to files
        stdout_file = NamedTemporaryFile(delete=False, mode='w+', dir=work_dir, prefix='stdout_', suffix='.log')
        stderr_file = NamedTemporaryFile(delete=False, mode='w+', dir=work_dir, prefix='stderr_', suffix='.log')

        process = Popen(
            request.command, 
            shell=True, 
            cwd=work_dir, 
            stdout=stdout_file,  # Redirect to temp file
            stderr=stderr_file,  # Redirect to temp file
            text=True
        )

        # Save job details in the jobs dictionary for later reference
        jobs[job_id] = {
            'process': process,
            'stdout_file': stdout_file.name,
            'stderr_file': stderr_file.name,
            'workdir': work_dir
        }

        return {"job_id": job_id, "status": "running"}

    # Handle synchronous (non-detached) execution
    else:
        process = Popen(
            request.command, 
            shell=True, 
            cwd=work_dir, 
            stdout=PIPE, 
            stderr=PIPE, 
            text=True
        )
        stdout, stderr = process.communicate()

        return {
            "stdout": stdout, 
            "stderr": stderr, 
            "exit_code": process.returncode, 
            "workdir": request.workdir
        }


@app.get("/subprocess/logs/{job_id}", response_model=dict, status_code=200, responses={
    200: {"description": "Real-time logs of the job"},
    404: {"description": "Job ID not found or logs unavailable"},
    401: {"description": "Unauthorized - authentication required"}
})
def get_job_logs(job_id: str, _ = Depends(user_auth_dependency)):
    """
    Fetch real-time logs for the specified job.
    
    - Returns 200 with stdout and stderr logs.
    - Returns 404 if the job ID is not found or logs are unavailable.
    """
    if job_id in jobs:
        job_info = jobs[job_id]
        stdout_file_path = job_info['stdout_file']
        stderr_file_path = job_info['stderr_file']
        
        stdout = ""
        stderr = ""

        # Read the logs from the stdout and stderr files
        if os.path.exists(stdout_file_path):
            with open(stdout_file_path, 'r') as f:
                stdout = f.read()
        else:
            print("Couldnt find stdout file")

        if os.path.exists(stderr_file_path):
            with open(stderr_file_path, 'r') as f:
                stderr = f.read()
        else:
            print("Couldnt find stderr file")

        # Check if the process is still running
        process = job_info['process']
        status = "running" if process.poll() is None else "completed"

        return {
            "stdout": stdout,
            "stderr": stderr,
            "status": status
        }
    else:
        raise HTTPException(status_code=404, detail="Job ID not found or logs unavailable")


@app.delete("/subprocess/terminate/{job_id}", response_model=dict, status_code=200, responses={
    200: {"description": "Subprocess successfully terminated"},
    404: {"description": "Job ID not found"},
    401: {"description": "Unauthorized - authentication required"}
})
def terminate_subprocess(job_id: str, _ = Depends(user_auth_dependency)):
    """
    Terminate the specified subprocess and clean up associated resources.
    
    - Returns 200 if the job is successfully terminated.
    - Returns 404 if the job ID is not found.
    """
    if job_id in jobs:
        process = jobs[job_id]['process']
        process.terminate()  # Sends SIGTERM

        # Cleanup stdout and stderr log files
        stdout_file = jobs[job_id]['stdout_file']
        stderr_file = jobs[job_id]['stderr_file']
        if os.path.exists(stdout_file):
            os.remove(stdout_file)
        if os.path.exists(stderr_file):
            os.remove(stderr_file)

        # Cleanup temporary directory if used
        workdir = jobs[job_id]['workdir']
        if workdir in temp_dirs:
            temp_dirs[workdir].cleanup()
            del temp_dirs[workdir]

        del jobs[job_id]
        return {"job_id": job_id, "status": "terminated"}
    else:
        raise HTTPException(status_code=404, detail="Job ID not found")

@app.get("/subprocess/status/{job_id}", response_model=JobStatus, status_code=200, responses={
    200: {"description": "Job status fetched successfully"},
    404: {"description": "Job ID not found"},
    401: {"description": "Unauthorized - authentication required"}
})
def get_job_status(job_id: str, _ = Depends(user_auth_dependency)):
    """
    Fetch the current status of the specified job, including completion status and output.
    
    - Returns 200 with job status.
    - Returns 404 if the job ID is not found.
    - Returns 401 if unauthorized.
    """
    if job_id in jobs:
        process = jobs[job_id]['process']
        if process.poll() is not None:  # Process has completed
            stdout, stderr = process.communicate()
            workdir = jobs[job_id]['workdir']

            # Cleanup temporary directory if used
            if workdir in temp_dirs:
                temp_dirs[workdir].cleanup()
                del temp_dirs[workdir]

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

@app.get("/session/{coding_session_id}/files/{file_path:path}", response_model=None, status_code=200, responses={
    200: {"description": "File served successfully"},
    404: {"description": "Session or file not found"},
    401: {"description": "Unauthorized - authentication required"}
})
async def get_file(coding_session_id: str, file_path: str, _ = Depends(user_auth_dependency)):
    """
    Serve a file from the session's working directory.
    
    - Returns 200 if the file is served successfully.
    - Returns 404 if the session or file is not found.
    - Returns 401 if unauthorized.
    """
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