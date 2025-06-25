import logging
import os
import re
import time
import atexit
import requests
from hashlib import md5
from typing import List, Union, Dict, ClassVar, Optional
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
from autogen.coding.utils import silence_pip
from autogen.code_utils import _cmd
from autogen.coding.local_commandline_code_executor import CommandLineCodeResult
from tools import MonsterNeoCodeRuntimeClient
from requests.exceptions import ConnectionError, Timeout

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommandLineCodeResultWithArtifact(CommandLineCodeResult):
    artifacts: Optional[List[str]]

class MonsterRemoteCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
    SUPPORTED_LANGUAGES: ClassVar[List[str]] = [
        "bash",
        "python"
    ]

    COMMON_SAVE_PATTERNS: ClassVar[Dict[str, str]] = {
        "matplotlib": r"plt\.savefig\(['\"](.*?)['\"]",  # Detect plt.savefig('filename')
        "opencv": r"cv2\.imwrite\(['\"](.*?)['\"]",      # Detect cv2.imwrite('filename')
        "numpy": r"np\.save\(['\"](.*?)['\"]",           # Detect np.save('filename')
        # Add other patterns if needed
    }

    def __init__(self, client: MonsterNeoCodeRuntimeClient, *args, **kwargs):
        """
        Initializes the remote command line executor with the MonsterNeoCodeRuntimeClient instance.
        
        Args:
            client (MonsterNeoCodeRuntimeClient): The client for interacting with the Monster runtime.
            *args, **kwargs: Other arguments passed to the parent LocalCommandLineCodeExecutor.
        """
        super().__init__(*args, **kwargs)
        self.client = client
        atexit.register(self.cleanup)

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

    def _write_code_remote(self, coding_session_id: str, filename: str, code: str) -> None:
        """
        Writes code to a file in the remote session's directory.

        Args:
            coding_session_id (str): The session ID to write the code in.
            filename (str): The name of the file to write.
            code (str): The code to write into the file.
        """
        logger.info(f"Writing code to remote server with filename: {filename}")
        response = self.client.session_manager.write_code(coding_session_id=coding_session_id, filename=filename, code=code)
        logger.info(f"Write Code Response: {response}")

    def _execute_remote(self, coding_session_id: str, command: str, detach: bool = False, workdir: str = None) -> Dict[str, Union[str, int]]:
        """
        Executes a command in the remote session.

        Args:
            coding_session_id (str): The session ID where the command will be executed.
            command (str): The command to execute.
            detach (bool): Whether to run the command in the background.
            workdir (str): The working directory for the command.

        Returns:
            Dict[str, Union[str, int]]: The result of the execution, including stdout, stderr, and exit code or job_id if detached.
        """
        logger.info(f"Executing command on remote server: {command} with detach={detach}")
        result = self.client.session_manager.run_subprocess(coding_session_id=coding_session_id, command=command, detach=detach, workdir=workdir)
        return result

    def _get_job_logs(self, job_id: str) -> Dict[str, str]:
        """
        Retrieves the logs of a running or completed job from the remote server.

        Args:
            job_id (str): The job ID of the process.
        
        Returns:
            Dict[str, str]: The logs, including stdout and stderr.
        """
        logger.info(f"Fetching logs for job_id: {job_id}")
        return self.client.session_manager.get_job_logs(job_id=job_id)

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
    
    def _find_output_files(self, code: str) -> List[str]:
        """
        Finds file paths that are likely to be saved in the code by searching for common
        file-saving functions such as plt.savefig, cv2.imwrite, np.save, etc.

        Args:
            code (str): The code block to search through.
        
        Returns:
            List[str]: A list of file paths that are detected in the code.
        """
        found_files = []
        for library, pattern in self.COMMON_SAVE_PATTERNS.items():
            matches = re.findall(pattern, code)
            if matches:
                logger.info(f"Detected {len(matches)} output files from {library} in code: {matches}")
                found_files.extend(matches)
        return found_files

    def _retrieve_output_files(self, session_id: str, output_files: List[str]) -> List[str]:
        """
        Retrieves files from the remote session and saves them locally.

        Args:
            session_id (str): The session ID from which to retrieve files.
            output_files (List[str]): List of file paths to retrieve.
            local_dir (str): The local directory to save the files in.

        Returns:
            List[str]: Paths of the files saved locally.
        """
        saved_files = []
        for file in output_files:
            local_path = os.path.join(self.work_dir, file)
            try:
                logger.info(f"Retrieving file {file} from session {session_id}")
                self.client.session_manager.get_file(session_id, file, local_path)
                saved_files.append(local_path)
                logger.info(f"File {file} saved successfully to {local_path}")
            except Exception as e:
                logger.error(f"Error retrieving file {file}: {e}")
        return saved_files

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CommandLineCodeResult:
        logs_all = ""  # Ensure logs_all is always a string
        exit_code = 0
        filename = None
        session_id = None
        error_output = ""  # Ensure error_output is always initialized
        try:
            # Retry mechanism for connection issues
            retries = 3
            while retries:
                try:
                    saved_files = []
                    # Create a new session
                    session_info = self.client.session_manager.create_session()
                    session_id = session_info["coding_session_id"]
                    logger.info(f"Session created with ID: {session_id}")
                    break
                except (ConnectionError, Timeout) as e:
                    retries -= 1
                    logger.error(f"Error creating session: {e}. Retrying {retries} more times...")
                    time.sleep(2)
                    if not retries:
                        raise
            
            for code_block in code_blocks:
                lang = self._detect_language(code_block.code)
                code = code_block.code

                # Detect potential output files
                detected_files = self._find_output_files(code)
                logger.info(f"Detected output files: {detected_files}")

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
                self._write_code_remote(coding_session_id=session_id, filename=filename, code=code)

                # Execute the code on the remote server
                result = self._execute_remote(coding_session_id=session_id, command=f"{_cmd(lang)} {filename}", detach=True)
                job_id = result.get("job_id")

                if job_id:
                    logger.info(f"Job {job_id} started, waiting for completion...")
                    # Poll for the job status and logs
                    while True:
                        status = self.client.session_manager.get_job_status(job_id)
                        logger.info(f"Job {job_id} status: {status['status']}")
                        if status['status'] == 'completed':
                            logs_all += status.get("stdout", "") or ""  # Avoid concatenating None
                            logs_all += status.get("stderr", "") or ""
                            exit_code = status.get("exit_code", 0)

                            if exit_code != 0:
                                error_output = self._parse_errors(logs_all, lang) or ""  # Ensure error_output is always a string
                                if error_output == 'No errors detected. Execution was successful.':
                                    error_output = "Most Probably is indent error! Please fix indent"
                    
                            logger.info(f"Job {job_id} completed with exit code {exit_code}\n detailed_status: {status}")
                            break
                        elif status['status'] == 'running':
                            logs = self._get_job_logs(job_id)
                            stdout_logs = logs.get("stdout", "") or ""  # Ensure no NoneType
                            stderr_logs = logs.get("stderr", "") or ""  # Ensure no NoneType
                            print(stdout_logs, end="")
                            print(stderr_logs, end="")
                            logs_all += stdout_logs + stderr_logs
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

                # Retrieve any detected files from the code
                if detected_files:
                    saved_files += self._retrieve_output_files(session_id, detected_files)

        except Exception as e:
            logger.error(f"Error during code execution: {e}")
            logs_all += f"Error: {str(e)}\n"
            exit_code = 1

        finally:
            if session_id:
                try:
                    self.client.session_manager.close_session(session_id)
                    logger.info(f"Session {session_id} closed successfully.")
                except Exception as e:
                    logger.error(f"Error closing session: {e}")

        return CommandLineCodeResultWithArtifact(exit_code=exit_code, output=logs_all + "\n" + error_output, code_file=filename, artifacts = saved_files)

    def cleanup(self):
        self.client.container_manager.terminate_container()


"" 
# Usage Example: Working
if __name__ == "__main__":
    # Initialize the client with the actual base URL and token
    client = MonsterNeoCodeRuntimeClient(container_type = "gpu")

    try:

        executor = MonsterRemoteCommandLineCodeExecutor(client=client)
        gpu_list = """
#!/bin/bash
nvidia-smi
"""

        dep_installation = """
#!/bin/bash
pip install scikit-learn numpy matplotlib opencv-python
"""

        long_running_python_code = """
import matplotlib.pyplot as plt
# Generate mock data
x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]

# Create a plot
plt.plot(x, y)

if True:
    print(True)

# Save the plot to a file
plt.savefig('plot.png')
print('Plot saved as plot.png')
"""

        gpu_example_code = """import cv2  # OpenCV
import numpy as np
import matplotlib.pyplot as plt

# Check if OpenCV was built with CUDA support
if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    print("No CUDA-enabled GPU found.")
    exit(200)

# Generate a mock image (simple gradient image)
mock_image = np.zeros((512, 512), dtype=np.uint8)
cv2.rectangle(mock_image, (100, 100), (400, 400), (255), thickness=-1)  # Draw a white square

# Upload the mock image to GPU
gpu_image = cv2.cuda_GpuMat()
gpu_image.upload(mock_image)

# Apply Gaussian blur on the GPU
blurred_gpu_image = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (15, 15), 0)
gpu_blurred = blurred_gpu_image.apply(gpu_image)

# Download the processed image back to the CPU
blurred_image = gpu_blurred.download()

# Plot the original and blurred images using Matplotlib
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.title('Original Mock Image')
#plt.imshow(mock_image, cmap='gray')
plt.axis('off')

# Blurred image
plt.subplot(1, 2, 2)
plt.title('Blurred Image (GPU)')
#plt.imshow(blurred_image, cmap='gray')
plt.axis('off')

# Save the plot to a file
plt.savefig('gpu_mock_image_processing_output.png')
print('Plot saved as gpu_mock_image_processing_output.png')
"""


        result = executor.execute_code_blocks([CodeBlock(code=dep_installation, language="bash"), CodeBlock(code = gpu_example_code, language="python"), CodeBlock(code=long_running_python_code, language="python")])
        logger.info(f"Final Output:\n{100*'#'}\n{result.output}")
        logger.info(f"Saved Files: {result.artifacts}")
        import pdb;pdb.set_trace()

    except Exception as e:
        import traceback;traceback.print_exc()
