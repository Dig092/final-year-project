import asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
import logging
import socket
import docker
import psutil
import pynvml
import httpx
import uuid
import os

from auth_utils import user_auth_dependency

from docker.models.containers import Container

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

app = FastAPI()
client = docker.from_env()

# Initialize NVML for GPU information
try:
    pynvml.nvmlInit()
    print(100*'^')
    print("GPU Found successfully!")
except Exception as e:
    pynvml = None
    print(100*'^')
    print("GPUs Not found!")

print(100*'^')

#################################################
#################################################
# Here session id is container runtime session id
#################################################
#################################################

sessions = {}  # session_id -> session details
gpu_usage = {}  # GPU index -> session_id

user_id_to_session_id = {}

host_ip = os.environ.get("HOST_IP", "localhost")

class ContainerRequest(BaseModel):
    """
    Request model for creating a new Docker container session.

    Attributes:
    - `type`: The type of container to create, either 'cpu' or 'gpu'. Defaults to 'cpu'.
    - `cpu_count`: The number of CPU cores to allocate for the container. Defaults to 1.
    - `memory`: The amount of memory (in GB) to allocate to the container. Defaults to 1 GB.
    - `shm_size`: Shared memory size for the container, typically for inter-process communication. Defaults to 2 GB.
    - `image`: The Docker image to use for the container. Defaults to 'qblockrepo/neo_agent_worker:cpu-latest'.
    """
    type: Literal["cpu", "gpu"] = Field("cpu", description="Type of container: 'cpu' or 'gpu'. Defaults to 'cpu'.")
    cpu_count: int = Field(1, description="Number of CPU cores to allocate. Defaults to 1.")
    memory: float = Field(1, description="Amount of memory to allocate (in GB). Defaults to 1 GB.")
    shm_size: str = Field("8gb", description="Shared memory size for the container. Default is '2gb'.")
    image: Literal["qblockrepo/neo_agent_worker:cpu-latest", "qblockrepo/neo_agent_worker:gpu-latest"] = Field(
        "qblockrepo/neo_agent_worker:cpu-latest", description="Docker image to use. Defaults to the CPU-latest image."
    )

class ContainerUtilization(BaseModel):
    """
    Response model for container utilization data.

    Attributes:
    - `cpu_percent`: The percentage of CPU usage.
    - `memory_usage_gb`: The amount of memory used by the container in GB.
    - `memory_percent`: The percentage of memory usage compared to the limit.
    - `user_id`: The user ID associated with the container.
    """
    cpu_percent: float = Field(..., description="CPU usage percentage of the container.")
    memory_usage_gb: float = Field(..., description="Memory usage in GB by the container.")
    memory_percent: float = Field(..., description="Percentage of memory used compared to the allocated limit.")
    user_id: str = Field(..., description="User ID associated with the container session.")

class ContainerLogs(BaseModel):
    """
    Response model for container logs.

    Attributes:
    - `logs`: The logs of the Docker container.
    - `user_id`: The user ID associated with the container.
    """
    logs: str = Field(..., description="Logs from the container.")
    user_id: str = Field(..., description="User ID associated with the container session.")

class ConnectedEndpointResponse(BaseModel):
    """
    Response model for connected endpoint information.

    Attributes:
    - `user_id`: The user ID associated with the container session.
    - `connected_endpoint`: The public endpoint (IP and port) of the running container.
    - `auth_token`: Authentication token for the container session.
    """
    user_id: str = Field(..., description="User ID associated with the container session.")
    connected_endpoint: str = Field(..., description="Public endpoint (IP and port) of the running container.")
    auth_token: str = Field(..., description="Authentication token for the container session.")

class TerminateContainerResponse(BaseModel):
    """
    Response model for terminating a container session.

    Attributes:
    - `terminated_containers`: List of terminated container session IDs.
    - `status`: Status of the termination request.
    """
    terminated_containers: list[str] = Field(..., description="List of terminated container session IDs.")
    status: str = Field(..., description="Status of the termination request.")

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

def find_available_port_restricted():
    predefined_ports = [8798, 8799, 8802, 8805]
    for port in predefined_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port  # This port is available
            except socket.error as e:
                logger.info(f"Port {port} is not available: {e}")
    raise HTTPException(status_code=503, detail="Max runtimes established for workers, please contact support at support@monsterapi.ai if issue persists!")


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


@app.post("/containers", response_model=dict, status_code=201, responses={
    201: {"description": "Container successfully created and resources allocated."},
    400: {"description": "Insufficient resources to allocate container."},
    503: {"description": "No available ports or maximum container limits reached."}
})
async def create_container(request: ContainerRequest, user_info=Depends(user_auth_dependency)):
    """
    Create a new Docker container session for either CPU or GPU workloads.

    The request defines the type of container, required resources (CPU, memory), and the Docker image to use.

    - `type`: 'cpu' or 'gpu' to specify the runtime environment.
    - `cpu_count`: Number of CPUs to allocate.
    - `memory`: Amount of memory to allocate (in GB).
    - `image`: Docker image to use for the container (CPU or GPU version).

    Returns a JSON response with:
    - `container_id`: The ID of the created container.
    - `status`: The creation status.
    - `gpu_index`: The index of the allocated GPU (if applicable).
    - `host_port`: The allocated host port for the container.
    - `connected_endpoint`: The public endpoint for accessing the container.
    - `auth_token`: The authentication token for the session.
    """
    if user_info["user_id"] in user_id_to_session_id:
        existing_session_id = user_id_to_session_id[user_info["user_id"]]
        existing_session = sessions[existing_session_id]
        return {
            "error": "User already has an active session.",
            "container_id": existing_session_id,
            "connected_endpoint": f"http://{host_ip}:{existing_session['resources']['host_port']}",
            "auth_token": "afcd6dd3-5657-4331-88f8-521f6569235d"
        }

    resource_index = check_resources(request)
    if resource_index == -1:
        raise HTTPException(status_code=400, detail="Insufficient resources")

    # Find an available port on the host
    available_port = find_available_port()

    gpu_index = str(resource_index) if request.type == 'gpu' and resource_index >= 0 else None

    if request.type == "gpu":
        gpu_capabilities = [['gpu']]
        device_request = docker.types.DeviceRequest(
            capabilities=gpu_capabilities,
            device_ids=[gpu_index] if gpu_index is not None else None
        )
    else:
        device_request = None

    container_name = uuid.uuid4()

    container = client.containers.run(
        request.image,
        name=container_name,
        detach=True,
        shm_size=request.shm_size,
        cpuset_cpus=str(request.cpu_count),
        mem_limit=f"{request.memory}g",
        ports={8000: available_port},
        init=True,
        device_requests=[device_request] if device_request else [],
        network="my_network"
    )

    session_id = str(container.id)
    sessions[session_id] = {
        'container': container,
        'last_active': datetime.now(),
        'container_name': container_name,
        'user_id': user_info["user_id"],
        'resources': {
            'type': request.type,
            'cpu_count': request.cpu_count,
            'memory': request.memory,
            'gpu_index': gpu_index,
            'host_port': available_port
        }
    }

    user_id_to_session_id[user_info["user_id"]] = session_id

    if gpu_index is not None:
        gpu_usage[int(gpu_index)] = session_id

    log_gpu_usage(user_info["user_id"], session_id, gpu_index)

    return {
        "container_id": container.id,
        "status": "created",
        "gpu_index": gpu_index,
        "host_port": available_port,
        "connected_endpoint": f"http://{host_ip}:{available_port}",
        "auth_token": "afcd6dd3-5657-4331-88f8-521f6569235d"
    }



@app.get("/containers/endpoint", response_model=ConnectedEndpointResponse, status_code=200, responses={
    200: {"description": "Successfully retrieved connected endpoint."},
    404: {"description": "No active container found for the user."}
})
async def get_connected_endpoint(user_info=Depends(user_auth_dependency)):
    """
    Retrieve the connected endpoint for the current user's active Docker container session.

    If the user has an active session, it returns the public IP address and port where the container is running.

    Response contains:
    - `user_id`: The user ID associated with the container session.
    - `connected_endpoint`: The public endpoint (IP and port) of the running container.
    - `auth_token`: Authentication token for the session.
    """
    user_id = user_info["user_id"]
    if user_id not in user_id_to_session_id:
        raise HTTPException(status_code=404, detail="No containers found for the specified user ID.")

    session_id = user_id_to_session_id[user_id]
    session = sessions[session_id]
    host_port = session['resources']['host_port']

    public_ip = host_ip if host_ip != "localhost" else socket.gethostbyname(socket.gethostname())

    return {
        "user_id": user_id,
        "connected_endpoint": f"http://{public_ip}:{host_port}",
        "auth_token": "afcd6dd3-5657-4331-88f8-521f6569235d"
    }



@app.get("/containers/utilization", response_model=ContainerUtilization, status_code=200, responses={
    200: {"description": "Successfully retrieved container utilization data."},
    404: {"description": "No active containers found for the user."}
})
async def get_container_utilization(user_info=Depends(user_auth_dependency)):
    """
    Get real-time resource utilization for the current user's active Docker container session.

    Returns CPU and memory usage as a percentage.

    Response contains:
    - `cpu_percent`: CPU usage percentage of the container.
    - `memory_usage_gb`: Memory usage in GB by the container.
    - `memory_percent`: Percentage of memory used compared to the allocated limit.
    - `user_id`: The user ID associated with the container session.
    """
    utilization = {}
    user_id = user_info["user_id"]
    if user_id not in user_id_to_session_id:
        raise HTTPException(status_code=404, detail="No containers found for the specified user ID.")
    session_id = user_id_to_session_id[user_id]
    session = sessions[session_id]
    container = session['container']
    stats = container.stats(stream=False)

    # Calculate CPU and memory usage
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

    return utilization[session_id]


@app.get("/containers/logs", response_model=ContainerLogs, status_code=200, responses={
    200: {"description": "Successfully retrieved container logs."},
    404: {"description": "No active containers found for the user."}
})
async def get_container_logs(
    lines: int = Query(10, description="Number of lines to tail from logs"), 
    user_info=Depends(user_auth_dependency)
):
    """
    Retrieve logs from the current user's active Docker container session.

    You can specify how many lines of logs to return using the `lines` query parameter.

    Response contains:
    - `logs`: The container's log output.
    - `user_id`: The user ID associated with the container session.
    """
    logs = {}
    user_id = user_info["user_id"]
    if user_id not in user_id_to_session_id:
        raise HTTPException(status_code=404, detail="No containers found for the specified user ID.")

    session_id = user_id_to_session_id[user_id]
    session = sessions[session_id]
    container = session['container']
    container_logs = container.logs(tail=lines).decode('utf-8')  # Tail the last N lines
    logs[session_id] = {
        'logs': container_logs,
        'user_id': session['user_id']
    }

    return logs[session_id]


def get_container_ip(container: Container, network_name: str = "bridge"):
    """
    Retrieves the IP address of the container within a specified Docker network.
    Defaults to the 'bridge' network if no network name is provided.
    Handles potential exceptions when network details are missing.
    """
    try:
        container.reload()  # Ensure the container's information is up-to-date
        return container.attrs['NetworkSettings']['Networks'][network_name]['IPAddress']
    except KeyError as e:
        logger.error(f"Network {network_name} not found in container settings: {e}")
        raise HTTPException(status_code=500, detail=f"Network configuration error: {network_name} not found")

def get_container_name(container: Container):
    """
    Retrieves the name of the container.
    Handles potential exceptions when container details are incomplete.
    """
    try:
        container.reload()  # Ensure the container's information is up-to-date
        # Container names are stored in a list, get the first name
        return container.attrs['Name'].strip('/')
    except KeyError as e:
        logger.error(f"Container name not found in container settings: {e}")
        raise HTTPException(status_code=500, detail="Container name configuration error.")


@app.delete("/containers", response_model=TerminateContainerResponse, status_code=200, responses={
    200: {"description": "Container session successfully terminated."},
    404: {"description": "No active container session found for the user."}
})
async def terminate_container(user_info=Depends(user_auth_dependency)):
    """
    Terminate the current user's active Docker container session.

    This route forcibly removes the Docker container associated with the user and cleans up the session.
    
    - If the container uses GPU resources, the GPU is freed upon termination.
    
    Response contains:
    - `terminated_containers`: List of terminated container session IDs.
    - `status`: Status of the termination request.
    """
    user_id = user_info["user_id"]
    terminated_containers = []
    
    # Check if user has an active session
    if user_id not in user_id_to_session_id:
        raise HTTPException(status_code=404, detail="No containers found for the specified user ID.")
    
    session_id = user_id_to_session_id[user_id]
    session = sessions[session_id]
    container = session['container']
    
    # Forcefully remove the container
    container.remove(force=True)
    terminated_containers.append(session_id)
    
    # Free GPU resources if used
    if 'gpu_index' in session['resources'] and session['resources']['gpu_index'] is not None:
        log_entry = f"GPU {session['resources']['gpu_index']} freed up from session {session_id}"
        logger.info(log_entry)
        del gpu_usage[int(session['resources']['gpu_index'])]
    
    # Clean up session and mappings
    del sessions[session_id]
    del user_id_to_session_id[user_id]
    
    # Return response with terminated containers and status
    return {"terminated_containers": terminated_containers, "status": "terminated"}

def log_gpu_usage(user_id, session_id, gpu_index):
    port = sessions[session_id]['resources']['host_port']
    resource_type = 'GPU' if gpu_index else 'CPU'
    log_entry = f"User {user_id} allocated {resource_type} resources in session {session_id} on port {port}"
    logger.info(log_entry)

async def cleanup_containers():
    while True:
        current_time = datetime.now()
        logger.info("Starting periodic container cleanup")
        
        for session_id in list(sessions.keys()):
            try:
                session = sessions[session_id]
                if (current_time - session['last_active']) > timedelta(minutes=3000):
                    logger.info(f"Cleaning up inactive session {session_id}")
                    
                    # Remove container
                    try:
                        session['container'].remove(force=True)
                        logger.info(f"Container for session {session_id} removed successfully")
                    except docker.errors.NotFound:
                        logger.warning(f"Container for session {session_id} not found, may have been removed externally")
                    except Exception as e:
                        logger.error(f"Error removing container for session {session_id}: {str(e)}")
                    
                    # Free GPU resources if used
                    if 'gpu_index' in session['resources'] and session['resources']['gpu_index'] is not None:
                        gpu_index = int(session['resources']['gpu_index'])
                        if gpu_index in gpu_usage:
                            del gpu_usage[gpu_index]
                            logger.info(f"GPU {gpu_index} freed up from session {session_id}")
                    
                    # Clean up user mapping
                    user_id = session['user_id']
                    if user_id in user_id_to_session_id and user_id_to_session_id[user_id] == session_id:
                        del user_id_to_session_id[user_id]
                        logger.info(f"User mapping removed for user {user_id}")
                    
                    # Remove session
                    del sessions[session_id]
                    logger.info(f"Session {session_id} removed from active sessions")
            
            except Exception as e:
                logger.error(f"Unexpected error during cleanup of session {session_id}: {str(e)}")
        
        logger.info("Periodic container cleanup completed")
        await asyncio.sleep(300)  # Run every 5 minutes

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_containers())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
