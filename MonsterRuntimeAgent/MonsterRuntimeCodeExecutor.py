import logging
import os
import re
import time
import requests
from hashlib import md5
from typing import List, Union, Dict, ClassVar, Optional

from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
from autogen.coding.utils import silence_pip
from autogen.code_utils import _cmd
from autogen.coding.local_commandline_code_executor import CommandLineCodeResult

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MonsterRemoteCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
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

        if self.auth_token is None:
            raise ValueError("Please provide auth_token arg or set MONSTER_API_KEY_NEO env var!")
        
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
        logger.info(f"Writing code to remote server with filename: {filename}")
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        workdir = response.json().get("workdir")
        logger.info(f"Code written to remote server at workdir: {workdir}")
        return workdir

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
        logger.info(f"Executing command on remote server: {command} with detach={detach}")
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        if detach:
            logger.info(f"Detached execution started with job_id: {result.get('job_id')}")
        return result

    def _get_job_status(self, job_id: str) -> Dict[str, Union[str, int]]:
        """
        Retrieves the status of a running job from the remote server.

        Args:
            job_id (str): The job ID of the running process.
        
        Returns:
            Dict[str, Union[str, int]]: The job status, including stdout, stderr, and exit code.
        """
        url = f"{self.remote_url}/subprocess/status/{job_id}"
        logger.info(f"Fetching status for job_id: {job_id}")
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
        logger.info(f"Fetching logs for job_id: {job_id}")
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def _detect_language(self, code: str) -> str:
        """
        Detects the programming language of the provided code.

        Args:
            code (str): The code block whose language needs to be detected.

        Returns:
            str: Detected language.
        """
        # Detect Python
        if re.search(r'^\s*import\s|\s*def\s|\s*class\s|\s*print\(', code, re.MULTILINE):
            return "python"
        
        # Detect Bash
        if re.search(r'^\s*#!/bin/bash\s|^\s*echo\s|^\s*ls\s|^\s*cd\s', code, re.MULTILINE):
            return "bash"

        # Fallback to using shebang if present
        if code.startswith("#!"):
            if "python" in code.lower():
                return "python"
            elif "bash" in code.lower():
                return "bash"

        logger.warning("Unable to detect language, defaulting to python.")
        return "python"

    def _parse_errors(self, logs: str, language: str) -> str:
        """
        Parses the logs for errors specific to the language.

        Args:
            logs (str): The logs to parse.
            language (str): The programming language of the code.

        Returns:
            str: Parsed errors or a success message.
        """
        errors = []
        
        if language == "python":
            # Look for common Python error patterns
            error_patterns = [r'Traceback \(most recent call last\):', r'Error:', r'Exception:']
            for pattern in error_patterns:
                match = re.search(pattern, logs)
                if match:
                    errors.append(logs[match.start():])
                    break
        
        elif language == "bash":
            # Look for common Bash error patterns in stderr
            error_patterns = [r'command not found', r'No such file or directory', r'permission denied']
            for pattern in error_patterns:
                match = re.search(pattern, logs, re.IGNORECASE)
                if match:
                    errors.append(logs[match.start():].splitlines()[0])
                    break

        if errors:
            return "\n".join(errors)
        else:
            return "No errors detected. Execution was successful."

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CommandLineCodeResult:
        logs_all = ""
        exit_code = 0
        filename = None

        for code_block in code_blocks:
            lang = self._detect_language(code_block.code)
            code = code_block.code
            if "pip" in code:
                code = code.replace("!pip", "pip")
                code = silence_pip(code, lang)
                lang = "bash"

            if lang not in self.SUPPORTED_LANGUAGES:
                logger.error(f"Unsupported language: {lang}")
                return CommandLineCodeResult(exit_code=1, output=f"Unsupported language: {lang}")

            code_hash = md5(code.encode()).hexdigest()
            filename = f"tmp_code_{code_hash}.{'py' if lang.startswith('python') else lang}"

            logger.info(f"Processing code block for language: {lang} with filename: {filename}")

            # Write the code to the remote server
            workdir = self._write_code_remote(filename, code)

            # Execute the code on the remote server
            result = self._execute_remote(f"{_cmd(lang)} {filename}", detach=True, workdir=workdir)
            job_id = result.get("job_id")

            if job_id:
                logger.info(f"Job {job_id} started, waiting for completion...")
                # Wait for the job to complete and retrieve logs
                while True:
                    status = self._get_job_status(job_id)
                    logger.info(f"Job {job_id} status: {status['status']}")
                    if status['status'] == 'completed':
                        logs_all += status.get("stdout", "") + status.get("stderr", "")
                        exit_code = status.get("exit_code", 0)
                        error_output = self._parse_errors(logs_all, lang)
                        logger.info(f"Job {job_id} completed with exit code {exit_code}\n detailed_status: {status}")
                        break
                    elif status['status'] == 'running':
                        logs = self._get_job_logs(job_id)
                        print("Logs:", logs)
                        print(logs.get("stdout", ""), end="")
                        print(logs.get("stderr", ""), end="")
                        logs_all += logs.get("stdout", "") + logs.get("stderr", "")
                    else:
                        logs_all += "Unexpected status received.\n"
                        logger.error(f"Job {job_id} encountered unexpected status: {status['status']}")
                        exit_code = 1
                        break
                    time.sleep(2)  # Polling interval to check job status
            else:
                logs_all += "Failed to start detached process.\n"
                logger.error("Failed to start detached process.")
                exit_code = 1
                break

        return CommandLineCodeResult(exit_code=exit_code, output=logs_all + "\n" + error_output, code_file=filename)

# Usage Example:
if __name__ == "__main__":
    executor = MonsterRemoteCommandLineCodeExecutor(remote_url="http://localhost:8000")

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
    logger.info(f"Final Output:\n{100*'#'}\n{result.output}")
