import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from tempfile import TemporaryDirectory
from subprocess import Popen, PIPE
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Global dictionary to manage jobs and temporary directories
jobs = {}  # job_id -> Popen object and metadata
temp_dirs = {}  # job_id -> PersistentTemporaryDirectory object

class JobStatus(BaseModel):
    job_id: str
    status: str  # 'running', 'completed', or 'failed'
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None

class CommandRequest(BaseModel):
    command: str
    detach: bool = False
    workdir: Optional[str] = None 

class CodeRequest(BaseModel):
    filename: str
    code: str
    workdir: Optional[str] = None  # Optional workdir for storing the code file

class PersistentTemporaryDirectory:
    """
    A wrapper around TemporaryDirectory that persists beyond the scope.
    The directory will not be deleted until explicitly done so.
    """
    def __init__(self):
        self._temp_dir = TemporaryDirectory()

    @property
    def name(self):
        return self._temp_dir.name

    def cleanup(self):
        """Manually clean up the temporary directory."""
        self._temp_dir.cleanup()

@app.post("/subprocess/write_code")
async def write_code(request: CodeRequest):
    try:
        if request.workdir:
            workdir = request.workdir
        else:
            temp_dir = PersistentTemporaryDirectory()
            workdir = temp_dir.name
            temp_dirs[workdir] = temp_dir  # Track the temp dir globally

        file_path = Path(workdir) / request.filename

        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        file_path.write_text(request.code)
        return {"file_path": str(file_path), "workdir": workdir, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write code to file: {str(e)}")

@app.post("/subprocess/run")
async def run_subprocess(request: CommandRequest):
    if request.workdir:
        workdir = request.workdir
    else:
        temp_dir = PersistentTemporaryDirectory()
        workdir = temp_dir.name
        temp_dirs[workdir] = temp_dir  # Track the temp dir globally

    if request.detach:
        # Run in the background and return immediately with a job_id
        process = Popen(request.command, shell=True, cwd=workdir, stdout=PIPE, stderr=PIPE, text=True)
        job_id = str(uuid.uuid4())
        jobs[job_id] = {'process': process, 'stdout': [], 'stderr': [], 'workdir': workdir}
        return {"job_id": job_id, "status": "Command running in background", "workdir": workdir}
    else:
        # Run synchronously and return output
        process = Popen(request.command, shell=True, cwd=workdir, stdout=PIPE, stderr=PIPE, text=True)
        stdout, stderr = process.communicate()

        # Cleanup the temporary directory if it was created for this job
        if workdir in temp_dirs:
            temp_dirs[workdir].cleanup()
            del temp_dirs[workdir]

        return {"stdout": stdout, "stderr": stderr, "exit_code": process.returncode, "workdir": workdir}

@app.get("/subprocess/logs/{job_id}")
def get_job_logs(job_id: str):
    if job_id in jobs and 'process' in jobs[job_id]:
        process_info = jobs[job_id]
        process = process_info['process']
        
        # If the process is still running, return whatever logs have been captured so far
        if process.poll() is None:
            # Attempt to read available output without blocking
            stdout, stderr = process.stdout.read(), process.stderr.read()
            if stdout:
                process_info['stdout'].append(stdout)
            if stderr:
                process_info['stderr'].append(stderr)
            return {"stdout": "".join(process_info['stdout']), "stderr": "".join(process_info['stderr'])}
        else:
            # If the process has completed, return all logs
            stdout, stderr = process.communicate()
            process_info['stdout'].append(stdout)
            process_info['stderr'].append(stderr)
            return {"stdout": "".join(process_info['stdout']), "stderr": "".join(process_info['stderr'])}
    else:
        raise HTTPException(status_code=404, detail="Job ID not found or logs unavailable")


@app.delete("/subprocess/terminate/{job_id}")
def terminate_subprocess(job_id: str):
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
def get_job_status(job_id: str):
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
