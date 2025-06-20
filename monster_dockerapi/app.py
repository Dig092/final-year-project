import asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Literal
import logging
import socket
import docker
import psutil
import pynvml
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

app = FastAPI()
client = docker.from_env()

# Initialize NVML for GPU information
try:
    pynvml.nvmlInit()
except Exception as e:
    pynvml = None

sessions = {}  # session_id -> session details
gpu_usage = {}  # GPU index -> session_id

user_id_to_session_id = {}

host_ip = os.environ.get("HOST_IP", "localhost")

class ContainerRequest(BaseModel):
    user_id: str = "vikas@qblocks.cloud"  # Unique identifier for the user
    type: Literal["cpu", "gpu"] = "cpu"   # 'cpu' or 'gpu'
    cpu_count: int = 1
    memory: float = 1  # Memory in GB
    shm_size: str = "2gb"  # Shared memory size, e.g., '2gb'
    image: str = "neo_docker_worker_cpu:latest"  # Docker image to run

def check_resources(request: ContainerRequest) -> int:
    available_cpus = psutil.cpu_count()
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    
    if request.type == 'gpu':
        if pynvml == None:
            raise HTTPException(status_code=404, detail="GPU Runtime Not supported!")
        available_gpus = pynvml.nvmlDeviceGetCount()
        for i in range(available_gpus):
            if i in gpu_usage:
                continue  # Skip if GPU is already in use
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if mem_info.free / (1024 ** 3) >= request.memory:  # Enough GPU memory available
                return i  # Return the index of the available GPU
        return -1  # No available GPU
    elif available_cpus >= request.cpu_count and available_memory >= request.memory:
        return -2  # Sufficient CPU and memory resources available
    return -1  # Insufficient resources

def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.

@app.post("/containers")
async def create_container(request: ContainerRequest):

    if request.user_id in user_id_to_session_id:
        existing_session_id = user_id_to_session_id[request.user_id]
        existing_session = sessions[existing_session_id]
        return { 
            "error": "User already has an active session.",
            "container_id": existing_session_id,
            "connected_endpoint": f"http://{host_ip}:{existing_session['resources']['host_port']}"
        }


    resource_index = check_resources(request)
    if resource_index == -1:
        raise HTTPException(status_code=400, detail="Insufficient resources")

    # Find an available port on the host
    available_port = find_available_port()

    gpu_index = str(resource_index) if request.type == 'gpu' and resource_index >= 0 else None

    if request.type == "gpu":
        gpu_capabilities = [['gpu']]
        device_ids = [gpu_index]
    else:
        gpu_capabilities = None
        device_ids = None

    #command = ["tail", "-f", "/dev/null"]
    
    container = client.containers.run(
        request.image,
        detach=True,
        shm_size=request.shm_size,
        cpuset_cpus=str(request.cpu_count),
        mem_limit=f"{request.memory}g",
        ports={8000: available_port},
        device_requests=[
            docker.types.DeviceRequest(count=1, capabilities=gpu_capabilities, device_ids=device_ids)
        ] if gpu_index is not None else []
    )

    session_id = str(container.id)
    sessions[session_id] = {
        'container': container,
        'last_active': datetime.now(),
        'user_id': request.user_id,
        'resources': {
            'type': request.type,
            'cpu_count': request.cpu_count,
            'memory': request.memory,
            'gpu_index': gpu_index,
            'host_port': available_port
        }
    }

    user_id_to_session_id[request.user_id] = session_id

    if gpu_index is not None:
        gpu_usage[int(gpu_index)] = session_id
    log_gpu_usage(request.user_id, session_id, gpu_index)

    return {"container_id": container.id, "status": "created", "gpu_index": gpu_index, "host_port": available_port}

@app.get("/containers/endpoint")
async def get_connected_endpoint(user_id: str = Query(..., description="User ID to retrieve the endpoint for")):
    session_id = user_id_to_session_id[user_id]
    session = sessions[session_id]
    host_port = session['resources']['host_port']
    return {
                "user_id": user_id,
                "connected_endpoint": f"http://{host_ip}:{host_port}"
            }
    raise HTTPException(status_code=404, detail="No active session found for the specified user ID.")

@app.get("/containers/utilization")
async def get_container_utilization(user_id: str = Query(..., description="User ID to filter containers")):
    utilization = {}
    session_id = user_id_to_session_id[user_id]
    session = sessions[session_id]
    container = session['container']
    stats = container.stats(stream=False)
    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
    system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
    if system_delta > 0:
        cpu_percent = (cpu_delta / system_delta) * stats['cpu_stats']['online_cpus'] * 100.0
    else:
        cpu_percent = 0.0

    mem_usage = stats['memory_stats']['usage'] / (1024 ** 3)  # Convert to GB
    mem_limit = stats['memory_stats']['limit'] / (1024 ** 3)  # Convert to GB
    mem_percent = (mem_usage / mem_limit) * 100.0

    utilization[session_id] = {
                'cpu_percent': cpu_percent,
                'memory_usage_gb': mem_usage,
                'memory_percent': mem_percent,
                'user_id': session['user_id']
            }
    if not utilization:
        raise HTTPException(status_code=404, detail="No containers found for the specified user ID.")
    return utilization

@app.get("/containers/logs")
async def get_container_logs(
    user_id: str = Query(..., description="User ID to filter containers"),
    lines: int = Query(10, description="Number of lines to tail from logs")
):
    logs = {}
    session_id = user_id_to_session_id[user_id]
    session = sessions[session_id]
    container = session['container']
    container_logs = container.logs(tail=lines).decode('utf-8')  # Tail the last N lines
    logs[session_id] = {
                'logs': container_logs,
                'user_id': session['user_id']
            }
    if not logs:
        raise HTTPException(status_code=404, detail="No containers found for the specified user ID.")
    return logs

@app.delete("/containers")
async def terminate_container(user_id: str = Query(..., description="User ID to filter containers")):
    terminated_containers = []
    session_id = user_id_to_session_id[user_id]
    session = sessions[session_id]
    container = session['container']
    container.remove(force=True)
    terminated_containers.append(session_id)
    
    if 'gpu_index' in session['resources'] and session['resources']['gpu_index'] is not None:
        log_entry = f"GPU {session['resources']['gpu_index']} freed up from session {session_id}"
        logger.info(log_entry)
        del gpu_usage[int(session['resources']['gpu_index'])]
    
    del sessions[session_id]
    
    if not terminated_containers:
        raise HTTPException(status_code=404, detail="No containers found for the specified user ID.")
    
    return {"terminated_containers": terminated_containers, "status": "terminated"}

def log_gpu_usage(user_id, session_id, gpu_index):
    port = sessions[session_id]['resources']['host_port']
    resource_type = 'GPU' if gpu_index else 'CPU'
    log_entry = f"User {user_id} allocated {resource_type} resources in session {session_id} on port {port}"
    logger.info(log_entry)

async def cleanup_containers():
    while True:
        current_time = datetime.now()
        for session_id in list(sessions.keys()):
            session = sessions[session_id]
            if current_time - session['last_active'] > timedelta(minutes=30):
                session['container'].remove(force=True)
                if 'gpu_index' in session['resources'] and session['resources']['gpu_index'] is not None:
                    log_entry = f"GPU {session['resources']['gpu_index']} freed up from session {session_id}"
                    logger.info(log_entry)
                    del gpu_usage[int(session['resources']['gpu_index'])]
                del sessions[session_id]
        await asyncio.sleep(600)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_containers())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
