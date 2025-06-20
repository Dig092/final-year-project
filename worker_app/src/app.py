import uuid
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from subprocess import Popen, PIPE
from typing import Optional

app = FastAPI()

# Data storage for subprocesses
jobs = {}  # job_id -> Popen object and metadata

class JobStatus(BaseModel):
    job_id: str
    status: str  # 'running', 'completed', or 'failed'
    exit_code: Optional[int] = None

class CommandRequest(BaseModel):
    command: str
    detach: bool = False

@app.post("/subprocess/run")
async def run_subprocess(request: CommandRequest):
    if request.detach:
        # Run in the background and return immediately with a job_id
        process = Popen(request.command, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        job_id = str(uuid.uuid4())
        jobs[job_id] = {'process': process, 'stdout': [], 'stderr': []}
        return {"job_id": job_id, "status": "Command running in background"}
    else:
        # Run synchronously and return output
        process = Popen(request.command, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        stdout, stderr = process.communicate()
        return {"stdout": stdout, "stderr": stderr, "exit_code": process.returncode}

@app.get("/subprocess/status/{job_id}")
def get_job_status(job_id: str):
    if job_id in jobs:
        process = jobs[job_id]['process']
        return JobStatus(
            job_id=job_id,
            status='running' if process.poll() is None else 'completed',
            exit_code=process.returncode
        )
    else:
        raise HTTPException(status_code=404, detail="Job ID not found")

@app.get("/subprocess/logs/{job_id}")
def get_job_logs(job_id: str):
    if job_id in jobs and 'process' in jobs[job_id]:
        process_info = jobs[job_id]
        stdout, stderr = process_info['process'].communicate()
        process_info['stdout'].append(stdout)
        process_info['stderr'].append(stderr)
        return {"stdout": stdout, "stderr": stderr}
    else:
        raise HTTPException(status_code=404, detail="Job ID not found or logs unavailable")

@app.delete("/subprocess/terminate/{job_id}")
def terminate_subprocess(job_id: str):
    if job_id in jobs:
        process = jobs[job_id]['process']
        process.terminate()  # Sends SIGTERM
        return {"job_id": job_id, "status": "terminated"}
    else:
        raise HTTPException(status_code=404, detail="Job ID not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)