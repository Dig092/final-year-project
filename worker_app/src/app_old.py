import uuid
import select 

from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from tempfile import TemporaryDirectory
from subprocess import Popen, PIPE
from pydantic import BaseModel
from typing import Optional

from auth import user_auth_dependency

from fastapi.responses import FileResponse

app = FastAPI()

# Global management for sessions and jobs
sessions = {}  # session_id -> list of directories
jobs = {}  # job_id -> Popen object and metadata
temp_dirs = {}  # job_id -> PersistentTemporaryDirectory object
MAX_SESSIONS = 5  # Limit to 5 sessions per user

class JobStatus(BaseModel):
    job_id: str
    status: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None

class CommandRequest(BaseModel):
    session_id: str
    command: str
    detach: bool = False
    workdir: Optional[str] = None

class CodeRequest(BaseModel):
    session_id: str
    filename: str
    code: str
    workdir: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: str

class PersistentTemporaryDirectory:
    def __init__(self, session_id):
        self._temp_dir = TemporaryDirectory()
        self.session_id = session_id

    @property
    def name(self):
        return self._temp_dir.name

    def cleanup(self):
        self._temp_dir.cleanup()

@app.post("/session/create")
def create_session(_ = Depends(user_auth_dependency)):
    session_id = str(uuid.uuid4())
    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(status_code=403, detail="Maximum session limit reached")
    
    # Create and track a new persistent temporary directory for this session
    temp_dir = PersistentTemporaryDirectory(session_id)
    sessions[session_id] = []
    temp_dirs[session_id] = temp_dir  # Link the temp directory with the session ID

    return {"session_id": session_id, "temp_dir": temp_dir.name}

@app.delete("/session/close/{session_id}")
def close_session(session_id: str, _ = Depends(user_auth_dependency)):
    if session_id in sessions:
        for workdir in sessions[session_id]:
            if workdir in temp_dirs:
                temp_dirs[workdir].cleanup()
                del temp_dirs[workdir]
        del sessions[session_id]
        return {"session_id": session_id, "status": "closed"}
    else:
        raise HTTPException(status_code=404, detail="Session ID not found")

@app.post("/subprocess/write_code")
async def write_code(request: CodeRequest, _ = Depends(user_auth_dependency)):
    if request.session_id not in sessions:
        raise HTTPException(status_code=400, detail = "invalid session id! Please create session first!")
    print(request.workdir)
    if request.workdir != None:
        if request.workdir not in sessions.get(request.session_id, []):
            raise HTTPException(status_code=400, detail="Invalid or missing workdir for this session!")
        else:
            work_dir = request.workdir
    else:
        work_dir = temp_dirs[request.session_id].name

    print("Work Dir", work_dir)
    
    try:
        file_path = Path(work_dir) / request.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(request.code)
        return {"file_path": str(file_path), "workdir": request.workdir, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write code to file: {str(e)}")

@app.post("/subprocess/run")
async def run_subprocess(request: CommandRequest, _ = Depends(user_auth_dependency)):
    if request.session_id not in sessions:
        raise HTTPException(status_code=400, detail = "invalid session id! Please create session first!")
    print(request.workdir)
    if request.workdir != None:
        if request.workdir not in sessions.get(request.session_id, []):
            raise HTTPException(status_code=400, detail="Invalid or missing workdir for this session!")
        else:
            work_dir = request.workdir
    else:
        work_dir = temp_dirs[request.session_id].name

    print("Work Dir", work_dir)

    process = Popen(
        request.command, 
        shell=True, 
        cwd=work_dir, 
        stdout=PIPE, 
        stderr=PIPE, 
        text=True,
        bufsize=1  # Line buffered
    )
    stdout, stderr = process.communicate()
    return {"stdout": stdout, "stderr": stderr, "exit_code": process.returncode, "workdir": request.workdir}


@app.get("/subprocess/logs/{job_id}")
def get_job_logs(job_id: str, _ = Depends(user_auth_dependency)):
    if job_id in jobs and 'process' in jobs[job_id]:
        process_info = jobs[job_id]
        process = process_info['process']
        
        # Check if process is still running
        if process.poll() is None:
            # Use select to check if there's data to read from stdout and stderr
            reads = [process.stdout, process.stderr]
            readable, _, _ = select.select(reads, [], [], 0.1)  # 0.1 seconds timeout

            if process.stdout in readable:
                stdout = process.stdout.readline()
                if stdout:
                    process_info['stdout'].append(stdout)

            if process.stderr in readable:
                stderr = process.stderr.readline()
                if stderr:
                    process_info['stderr'].append(stderr)

            return {
                "stdout": "".join(process_info['stdout']),
                "stderr": "".join(process_info['stderr']),
                "status": "running"
            }
        else:
            # If the process has completed, return all logs
            stdout, stderr = process.communicate()
            process_info['stdout'].append(stdout)
            process_info['stderr'].append(stderr)
            return {
                "stdout": "".join(process_info['stdout']),
                "stderr": "".join(process_info['stderr']),
                "status": "completed"
            }
    else:
        raise HTTPException(status_code=404, detail="Job ID not found or logs unavailable")



@app.delete("/subprocess/terminate/{job_id}")
def terminate_subprocess(job_id: str, _ = Depends(user_auth_dependency)):
    if job_id in jobs:
        process = jobs[job_id]['process']
        process.terminate()  # Sends SIGTERM

        # Cleanup temporary directory if used
        workdir = jobs[job_id]['workdir']
        if workdir in temp_dirs:
            temp_dirs[workdir].cleanup()
            del temp_dirs[workdir]

        del jobs[job_id]
        return {"job_id": job_id, "status": "terminated"}
    else:
        raise HTTPException(status_code=404, detail="Job ID not found")

@app.get("/subprocess/status/{job_id}")
def get_job_status(job_id: str, _ = Depends(user_auth_dependency)):
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

@app.get("/session/{session_id}/files/{file_path:path}")
async def get_file(session_id: str, file_path: str, _ = Depends(user_auth_dependency)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    work_dir = temp_dirs.get(session_id)
    if not work_dir:
        raise HTTPException(status_code=404, detail="Working directory not found for session")

    full_path = Path(work_dir.name) / file_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=full_path, filename=file_path.split("/")[-1])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)