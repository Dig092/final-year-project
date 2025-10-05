import requests
import logging
import atexit
import time
import os
from typing import Literal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Container Management Class
class ContainerManager:
    def __init__(self, base_url: str, token: str, container_type: Literal["cpu", "gpu"] = "cpu", cpu_count: int = 1, memory: int = 1,  container_id: str = None):
        """
        Initializes the container manager with the base URL, token, and container specs.
        :param base_url: The base URL of the API.
        :param token: The bearer token for authentication.
        :param container_type: 'cpu' or 'gpu' (default: 'cpu').
        :param cpu_count: Number of CPUs to allocate (default: 1).
        :param memory: Memory to allocate in GB (default: 1GB).
        """
        self.base_url = base_url
        self.token = token
        self.management_headers = {"Authorization": f"Bearer {self.token}"}

        if container_type == "cpu":
            self.base_url = f"https://8080-{container_id}.e2b.dev"
        elif container_type == "gpu":
            self.base_url = f"https://{container_id}.monsterapi.ai"

        self.runtime_info = {"connected_endpoint": self.base_url, "auth_token":"afcd6dd3-5657-4331-88f8-521f6569235d"}
        # self.create_container(container_type=container_type, cpu_count=cpu_count, memory=memory, image=self.image)

    def _handle_response(self, response):
        """
        Handles API response, raising an error for non-success statuses.
        :param response: The HTTP response object.
        :return: The JSON content of the response if successful.
        """
        if response.status_code in [200, 201]:
            return response.json()
        else:
            response.raise_for_status()

    def create_container(self, container_type: str = "cpu", cpu_count: int = 1, memory: int = 1, image: str = "qblockrepo/neo_agent_worker:cpu-latest"):
        """
        Creates a new Docker container.
        :param container_type: 'cpu' or 'gpu' (default: 'cpu').
        :param cpu_count: Number of CPUs to allocate (default: 1).
        :param memory: Memory in GB (default: 1GB).
        :param image: Docker image to use (default: CPU-latest image).
        """
        url = f"{self.base_url}/containers"
        payload = {"type": container_type, "cpu_count": cpu_count, "memory": memory, "image": image}
        response = requests.post(url, headers=self.management_headers, json=payload)
        self.runtime_info = self._handle_response(response)
        logger.info(f"Created container successfully: {self.runtime_info}")
        return self.runtime_info

    def get_connected_endpoint(self):
        """
        Gets the connected endpoint for the active container session.
        :return: JSON with the connected endpoint.
        """
        url = f"{self.base_url}/containers/endpoint"
        response = requests.get(url, headers=self.management_headers)
        return self._handle_response(response)

    def get_container_utilization(self):
        """
        Retrieves real-time CPU and memory utilization of the container.
        :return: JSON with CPU and memory usage.
        """
        url = f"{self.base_url}/containers/utilization"
        response = requests.get(url, headers=self.management_headers)
        return self._handle_response(response)

    def get_container_logs(self, lines: int = 10):
        """
        Retrieves logs from the container session.
        :param lines: Number of lines to tail from logs (default: 10).
        :return: JSON with container logs.
        """
        url = f"{self.base_url}/containers/logs?lines={lines}"
        response = requests.get(url, headers=self.management_headers)
        return self._handle_response(response)

    def terminate_container(self):
        """
        Terminates the active container session.
        :return: JSON with the termination status.
        """
        
        return {}

        url = f"{self.base_url}/containers"
        response = requests.delete(url, headers=self.management_headers)
        logger.info(f"Terminated container: {response.json()}")
        return self._handle_response(response)


# Session Management Class
class SessionManager:
    def __init__(self, runtime_info: dict):
        """
        Initializes the session manager with the runtime info and token.
        :param runtime_info: The runtime information from the container manager.
        """
        self.runtime_url = runtime_info["connected_endpoint"]
        self.token = runtime_info["auth_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def _handle_response(self, response):
        """
        Handles API response, raising an error for non-success statuses.
        :param response: The HTTP response object.
        :return: The JSON content of the response if successful.
        """
        if response.status_code in [200, 201]:
            return response.json()
        else:
            response.raise_for_status()

    def create_session(self):
        """
        Creates a new coding session.
        :return: The session information.
        """
        retries = 0
        while retries < 10:
            retries += 1
            try:
                url = f"{self.runtime_url}/session/create"
                response = requests.post(url, headers=self.headers, verify=False)
                print(url, response)
                return self._handle_response(response)
            except Exception as e:
                time.sleep(0.2)

        raise RuntimeError("Cannot Create Session!")

    def close_session(self, coding_session_id: str):
        """
        Closes a coding session by its ID.
        :param coding_session_id: The ID of the session to close.
        :return: Success message or error.
        """
        url = f"{self.runtime_url}/session/close/{coding_session_id}"
        response = requests.delete(url, headers=self.headers, verify=False)
        return self._handle_response(response)

    def delete_tmp(self, coding_session_id: str):
        """
        Cleans up tmpdir in sandbox.
        :param coding_session_id: The session ID.
        :return: Output or job details.
        """
        url = f"{self.runtime_url}/subprocess/run"
        payload = {"coding_session_id": coding_session_id, "command": "rm -rf /tmp/*", "detach": False, "workdir": "/"}
        response = requests.post(url, headers=self.headers, json=payload, verify=False)
        return self._handle_response(response)

    def write_code(self, coding_session_id: str, filename: str, code: str, workdir: str = None):
        """
        Writes code to a file in the session's working directory.
        :param coding_session_id: The session ID.
        :param filename: The name of the file to write.
        :param code: The code to write to the file.
        :param workdir: The working directory (optional).
        :return: Success message or error.
        """
        url = f"{self.runtime_url}/subprocess/write_code"
        payload = {"coding_session_id": coding_session_id, "filename": filename, "code": code, "workdir": workdir}
        response = requests.post(url, headers=self.headers, json=payload, verify=False)
        return self._handle_response(response)

    def run_subprocess(self, coding_session_id: str, command: str, detach: bool = False, workdir: str = None):
        """
        Runs a subprocess in the session.
        :param coding_session_id: The session ID.
        :param command: The command to execute.
        :param detach: Whether to run the process asynchronously.
        :param workdir: The working directory (optional).
        :return: Output or job details.
        """
        url = f"{self.runtime_url}/subprocess/run"
        payload = {"coding_session_id": coding_session_id, "command": command, "detach": detach, "workdir": workdir}
        response = requests.post(url, headers=self.headers, json=payload, verify=False)
        return self._handle_response(response)
    
    def get_file(self, coding_session_id: str, file_path: str, local_path: str):
        """
        Retrieves a file from the session's working directory and saves it locally.
        
        :param coding_session_id: The ID of the session.
        :param file_path: The path of the file within the session directory.
        :param local_path: The local path where the file will be saved.
        :return: A message indicating success or failure.
        """
        url = f"{self.runtime_url}/session/{coding_session_id}/files/{file_path}"

        # Send request to get the file from the session
        response = requests.get(url, headers=self.headers, verify=False)
        
        # Check if the file was successfully retrieved
        if response.status_code == 200:
            # Save the file to the specified local path
            with open(local_path, "wb") as file:
                file.write(response.content)
            return f"File saved successfully to {local_path}"
        elif response.status_code == 404:
            raise FileNotFoundError("File or session not found")
        elif response.status_code == 401:
            raise PermissionError("Unauthorized access to the session files")
        else:
            response.raise_for_status()

    def get_job_logs(self, job_id: str):
        """
        Fetches logs for a specific job.
        :param job_id: The job ID.
        :return: Logs or error if the job is not found.
        """
        url = f"{self.runtime_url}/subprocess/logs/{job_id}"
        response = requests.get(url, headers=self.headers, verify=False)
        return self._handle_response(response)

    def get_job_status(self, job_id: str):
        """
        Fetches the status of a job.
        :param job_id: The job ID.
        :return: Job status or error.
        """
        url = f"{self.runtime_url}/subprocess/status/{job_id}"
        response = requests.get(url, headers=self.headers, verify=False)
        return self._handle_response(response)

    def terminate_subprocess(self, job_id: str):
        """
        Terminates a running subprocess by its job ID.
        :param job_id: The job ID.
        :return: Success message or error.
        """
        url = f"{self.runtime_url}/subprocess/terminate/{job_id}"
        response = requests.delete(url, headers=self.headers, verify=False)
        return self._handle_response(response)


# Main Client Class
class MonsterNeoCodeRuntimeClient:
    def __init__(self, base_url: str = "http://localhost:8000", token: str = None, container_type: Literal["cpu", "gpu"] = "cpu", cpu_count: int = 1, memory: int = 1, container_id: str = None):
        """
        Initializes the Monster Neo Client with base URL and container/session management.
        """
        if not token:
            token = os.environ.get("MONSTER_API_KEY_NEO")
            if not token:
                raise RuntimeError("Please pass in token arg or set MONSTER_API_KEY_NEO env!")

        self.container_manager = ContainerManager(base_url, token, container_type, cpu_count, memory, container_id=container_id)
        self.session_manager = SessionManager(self.container_manager.runtime_info)
    
    def cleanup(self, session_id):
        try:
            self.session_manager.close_session(session_id)
        except Exception as e:
            pass
        
        self.container_manager.terminate_container()


if __name__ == "__main__":
    import time
    # Initialize the client
    client = MonsterNeoCodeRuntimeClient()
    
    # Step 1: Get connected endpoint
    print(client.container_manager.get_connected_endpoint())

    # Step 2: Check container logs and utilization
    print(client.container_manager.get_container_logs())
    print(client.container_manager.get_container_utilization())

    # Step 3: Create a new session
    session = client.session_manager.create_session()
    session_id = session["coding_session_id"]
    print(f"Session created with ID: {session_id}")

    # Step 4: Write code to install matplotlib and generate a mock plot
    code_to_plot = """
import matplotlib.pyplot as plt

# Generate mock data
x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]

# Create a plot
plt.plot(x, y)

# Save the plot to a file
plt.savefig('plot.png')
print('Plot saved as plot.png')
    """

    # Step 5: Install matplotlib in the container
    try:
        # Write the installation and plotting script
        response = client.session_manager.write_code(
            coding_session_id=session_id,
            filename="plot_script.py",
            code=code_to_plot
        )
        print(f"Write Code Response: {response}")
    except Exception as e:
        print(f"Error writing code: {e}")
        client.cleanup(session_id)
        exit(1)

    # Step 6: Run the script to install matplotlib and generate the plot
    try:
        output = client.session_manager.run_subprocess(coding_session_id=session_id, command="pip install matplotlib")
        print(f"Run Subprocess Output: {output['stdout']}")
    except Exception as e:
        print(f"Error running subprocess: {e}")
        client.cleanup(session_id)
        exit(1)

    # Step 6: Run the script to install matplotlib and generate the plot
    try:
        output = client.session_manager.run_subprocess(coding_session_id=session_id, command="python plot_script.py")
        print(f"Run Subprocess Output: {output['stdout']}")
    except Exception as e:
        print(f"Error running subprocess: {e}")
        client.cleanup(session_id)
        exit(1)

    # Step 7: Retrieve and save the generated plot file locally
    try:
        file_path = "plot.png"
        local_path = "downloaded_plot.png"  # Local path to save the file
        response = client.session_manager.get_file(coding_session_id=session_id, file_path=file_path, local_path=local_path)
        print(f"File Retrieved: {response}")
    except Exception as e:
        print(f"Error retrieving file: {e}")
        client.cleanup(session_id)
        exit(1)

    # Step 8: Close the session
    try:
        response = client.cleanup(session_id)
        print(f"Session closed: {response}")
    except Exception as e:
        print(f"Error closing session: {e}")

