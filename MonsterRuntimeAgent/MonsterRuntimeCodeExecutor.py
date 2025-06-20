import os
import time
import requests
from hashlib import md5
from pathlib import Path
from typing import List, Union, Dict, ClassVar, Optional

from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
from autogen.coding.utils import _get_file_name_from_content
from autogen.code_utils import _cmd
from autogen.coding.local_commandline_code_executor import CommandLineCodeResult

class RemoteCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
    SUPPORTED_LANGUAGES: ClassVar[List[str]] = [
        "bash",
        "python"
    ]

    def __init__(self, remote_url: str, auth_token: Optional[str] = None, *args, **kwargs):
        """
        Initializes the remote command line executor with the remote FastAPI URL.
        
        Args:
            remote_url (str): The URL of the FastAPI service hosting the subprocess routes.
            *args, **kwargs: Other arguments passed to the parent LocalCommandLineCodeExecutor.
        """
        super().__init__(*args, **kwargs)
        self.remote_url = remote_url
        self.auth_token = auth_token or os.getenv('MONSTER_API_KEY_NEO')

        if self.auth_token == None:
            raise ValueError("Please feed auth_token arg or set MONSTER_API_KEY_NEO env var!")
        
        self.headers = {
            'Authorization': f'Bearer {self.auth_token}',
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def _write_code_remote(self, filename: str, code: str) -> str:
        """
        Writes code to a file on the remote server.

        Args:
            filename (str): The name of the file to write.
            code (str): The code to write into the file.

        Returns:
            str: The working directory on the remote server where the file was written.
        """
        url = f"{self.remote_url}/subprocess/write_code"
        payload = {"filename": filename, "code": code}
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json().get("workdir")

    def _execute_remote(self, command: str, detach: bool = False, workdir: str = None) -> Dict[str, Union[str, int]]:
        """
        Executes a command on the remote server.

        Args:
            command (str): The command to execute.
            detach (bool): Whether to run the command in the background.
            workdir (str): The working directory for the command.
        
        Returns:
            Dict[str, Union[str, int]]: The result of the execution, including stdout, stderr, and exit code or job_id if detached.
        """
        url = f"{self.remote_url}/subprocess/run"
        payload = {"command": command, "detach": detach, "workdir": workdir}
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def _get_job_status(self, job_id: str) -> Dict[str, Union[str, int]]:
        """
        Retrieves the status of a running job from the remote server.

        Args:
            job_id (str): The job ID of the running process.
        
        Returns:
            Dict[str, Union[str, int]]: The job status, including stdout, stderr, and exit code.
        """
        url = f"{self.remote_url}/subprocess/status/{job_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def _get_job_logs(self, job_id: str) -> Dict[str, str]:
        """
        Retrieves the logs of a running or completed job from the remote server.

        Args:
            job_id (str): The job ID of the process.
        
        Returns:
            Dict[str, str]: The logs, including stdout and stderr.
        """
        url = f"{self.remote_url}/subprocess/logs/{job_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CommandLineCodeResult:
        logs_all = ""
        exit_code = 0

        for code_block in code_blocks:
            lang, code = code_block.language, code_block.code
            lang = lang.lower()

            if lang not in self.SUPPORTED_LANGUAGES:
                return CommandLineCodeResult(exit_code=1, output=f"Unsupported language: {lang}")

            filename = _get_file_name_from_content(code, self._work_dir)
            if filename is None:
                code_hash = md5(code.encode()).hexdigest()
                filename = f"tmp_code_{code_hash}.{'py' if lang.startswith('python') else lang}"

            # Write the code to the remote server
            workdir = self._write_code_remote(filename, code)

            # Execute the code on the remote server
            result = self._execute_remote(f"{_cmd(lang)} {filename}", detach=True, workdir=workdir)
            job_id = result.get("job_id")

            if job_id:
                # Wait for the job to complete and retrieve logs
                while True:
                    status = self._get_job_status(job_id)
                    if status['status'] == 'completed':
                        logs_all += status.get("stdout", "") + status.get("stderr", "")
                        exit_code = status.get("exit_code", 0)
                        break
                    elif status['status'] == 'running':
                        import pdb;pdb.set_trace()
                        logs = self._get_job_logs(job_id)
                        print(logs.get("stdout", ""), end="")
                        print(logs.get("stderr", ""), end="")
                        logs_all += logs.get("stdout", "") + logs.get("stderr", "")
                    else:
                        logs_all += "Unexpected status received.\n"
                        exit_code = 1
                        break
                    time.sleep(2)  # Polling interval to check job status
            else:
                logs_all += "Failed to start detached process.\n"
                exit_code = 1
                break

        return CommandLineCodeResult(exit_code=exit_code, output=logs_all, code_file=filename)

# Usage Example:
if __name__ == "__main__":
    executor = RemoteCommandLineCodeExecutor(remote_url="http://jus.qblocks.cloud:8002323")

    result = executor.execute_code_blocks([CodeBlock(code="echo 'Hello, World!'", language="bash")])
    print(result.output)
    
    # Python code that runs for about 2 minutes, printing output every 10 seconds
    long_running_python_code = """
import time

for i in range(1, 13):  # Run for 12 iterations (2 minutes)
    print(f"Progress: {i * 10}%")
    time.sleep(5)

print("Completed long-running task!")
"""

    # Execute the long-running Python code remotely
    result = executor.execute_code_blocks([CodeBlock(code=long_running_python_code, language="python")])

    # Print the final output after completion
    print(f"Final Output:\n{100*'#'}\n", result.output)

