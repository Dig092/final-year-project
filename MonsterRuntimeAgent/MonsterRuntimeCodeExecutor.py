import logging
import os
import re
import time
import uuid
import atexit
import requests
from hashlib import md5
from datetime import datetime
from typing import List, Union, Dict, ClassVar, Optional
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
from autogen.coding.utils import silence_pip
from autogen.code_utils import _cmd
from autogen.coding.local_commandline_code_executor import CommandLineCodeResult

from MonsterRuntimeAgent.Tools.RuntimeTools import MonsterNeoCodeRuntimeClient
from requests.exceptions import ConnectionError, Timeout

# Configure the logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

logfile = '/tmp/streamneo.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
file_handler = logging.FileHandler(logfile)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('Agent Actions: %(asctime)s - NeoCodingAgent - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Example logging
logger.info("This is an info message.")

def trim_logs(logs, max_size=30000):
    if len(logs) <= max_size:
        return logs
    
    # Keep the first and last 4000 characters
    keep_size = 5000
    start = logs[:keep_size]
    end = logs[-keep_size:]
    
    # Add a message in the middle indicating trimming
    middle_msg = f"\n... [Trimmed {len(logs) - 2*keep_size} characters] ...\n"
    
    return start + middle_msg + end

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
        # Add other patterns also !!!
    }

    def __init__(self, client: MonsterNeoCodeRuntimeClient, thread_id: str, *args, **kwargs):
        """
        Initializes the remote command line executor with the MonsterNeoCodeRuntimeClient instance.
        
        Args:
            client (MonsterNeoCodeRuntimeClient): The client for interacting with the Monster runtime.
            *args, **kwargs: Other arguments passed to the parent LocalCommandLineCodeExecutor.
        """
        super().__init__(*args, **kwargs)
        self.client = client
        self.thread_id = thread_id
        self._configure_logging()
        self.session_info = self.client.session_manager.create_session()
        self.job_list = []
        atexit.register(self.cleanup)

    def _configure_logging(self):
        """
        Configures a file-based logger specific to the thread ID.
        """
        log_filename = f"{self.thread_id}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def _extract_filename_from_code(self, code: str) -> Optional[str]:
        """
        Extracts the filename from the code if a filename comment is present.

        Args:
            code (str): The code block to search for a filename.

        Returns:
            Optional[str]: The extracted filename if found, otherwise None.
        """
        filename_match = re.search(r'#\s*Filename:\s*(\S+)', code)
        if filename_match:
            return filename_match.group(1)
        return None

    def _detect_language(self, code: str) -> Optional[str]:
        """
        Detects the programming language of the provided code using multiple heuristics.
        
        Args:
            code (str): The code block whose language needs to be detected.
            
        Returns:
            Optional[str]: Detected language ("python" or "bash") or None if unable to detect.
        """
        # Clean the code and ensure it's not empty
        if not code or not code.strip():
            return None
        
        code = code.strip()
        
        # Common patterns for each language
        python_patterns = {
            'imports': r'^\s*(?:from\s+[\w.]+\s+)?import\s+[\w.]+',
            'function_def': r'^\s*def\s+\w+\s*\([^)]*\)\s*:',
            'class_def': r'^\s*class\s+\w+',
            'print': r'print\s*\(',
            'python_builtins': r'(?:len|range|list|dict|set|tuple)\s*\(',
            'list_comprehension': r'\[\s*[\w\s.()]+\s+for\s+\w+\s+in\s+',
            'python_comments': r'^\s*#(?!\!)',  # Comments but not shebang
            'variable_assignment': r'^\s*\w+\s*=\s*[\w\'"{\[\(]',
            'pip_commands': r'^\s*!?pip\s+(?:install|uninstall|list)',
            'python_string': r'(?:f|r|b)?[\'"]{3}|(?:f|r|b)?[\'"]{1}',
            'indentation': r'^\s{4}\w+',
        }
        
        bash_patterns = {
            'command_chains': r'[|;&]',
            'file_operations': r'(?:^|\s)(?:cat|touch|rm|cp|mv|mkdir|chmod|chown)\s',
            'navigation': r'(?:^|\s)(?:cd|pwd|ls|find|grep)\s',
            'environment': r'(?:^|\s)(?:export|env|echo|source)\s',
            'package_management': r'(?:^|\s)(?:apt|yum|brew|npm|pip)\s',
            'process_management': r'(?:^|\s)(?:ps|kill|top|systemctl)\s',
            'redirections': r'[><]{1,2}',
            'variable_assignment': r'^\s*[\w]+=[\w\'"./]+',
            'for_loop': r'^\s*for\s+\w+\s+in\s+.+;\s*do',
            'if_statement': r'^\s*if\s+\[\[?.+\]\]?.+;\s*then',
            'shebang': r'^#!.*(?:bash|sh)',
            'bash_array': r'^\s*[\w]+=\(',
        }

        # Count matches for each language
        python_score = 0
        bash_score = 0

        # Check Python patterns
        for pattern in python_patterns.values():
            matches = len(re.findall(pattern, code, re.MULTILINE))
            python_score += matches

        # Check Bash patterns
        for pattern in bash_patterns.values():
            matches = len(re.findall(pattern, code, re.MULTILINE))
            bash_score += matches

        # Additional scoring based on specific characteristics
        
        # Python-specific characteristics
        if ':' in code and ('def' in code or 'class' in code):
            python_score += 3
        if 'import' in code.lower() and 'from' in code.lower():
            python_score += 2
        if re.search(r'^\s{4}', code, re.MULTILINE):  # Consistent 4-space indentation
            python_score += 2

        # Bash-specific characteristics
        if '$' in code and not '$$' in code:  # Variable usage in bash but not Python multiprocessing
            bash_score += 2
        if ';' in code and not re.search(r'for.*?;.*?do', code):  # Command separation
            bash_score += 1
        if re.search(r'(?<![\w])-[a-z]{1,2}\b', code):  # Command flags
            bash_score += 2

        # Special case for single-line commands
        if len(code.split('\n')) == 1:
            if re.match(r'^\s*!.*', code):  # Jupyter notebook shell command
                return "bash"
            if re.match(r'^\s*(?:pip|python|python3)\s', code):
                return "python"
            if re.match(r'^\s*[a-z]+(?:\s+-[a-zA-Z]+)*\s*', code):  # Typical bash command structure
                return "bash"

        # Make the final decision
        if python_score > bash_score:
            return "python"
        elif bash_score > python_score:
            return "bash"
        elif bash_score == python_score:
            # If scores are equal, look for definitive indicators
            if re.search(r'^\s*(?:def|class|import|from\s+\w+\s+import)', code, re.MULTILINE):
                return "python"
            elif re.search(r'^\s*(?:#!/bin/bash|#!/bin/sh)', code):
                return "bash"
            elif re.search(r'^\s*(?:cd|ls|mkdir|rm|cp|mv)\s', code, re.MULTILINE):
                return "bash"
        
        return None

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
        session_id = self.session_info["coding_session_id"]
        error_output = ""  # Ensure error_output is always initialized
        try:
            # Retry mechanism for connection issues
            retries = 3
            while retries:
                try:
                    saved_files = []
                    # Create a new session
                    #session_info = self.client.session_manager.create_session()
                    #session_id = self.session_info["coding_session_id"]
                    logger.info(f"Attaching to session created with ID: {session_id}")
                    break
                except (ConnectionError, Timeout) as e:
                    retries -= 1
                    logger.error(f"Error creating session: {e}. Retrying {retries} more times...")
                    time.sleep(2)
                    if not retries:
                        raise
            
            for code_block in code_blocks:
                lang = self._detect_language(code_block.code)
                if lang == None:
                    # error = """
                    # Couldnt detect either bash or python code.
                    # please start with shebang if bash.
                    # """
                    error = """Unable to definitively detect code type. Please ensure code contains clear Python or Bash characteristics."""
                    logger.error(error)
                    return CommandLineCodeResult(exit_code=1, output=error)

                filename = self._extract_filename_from_code(code_block.code)
                code = code_block.code

                # Detect potential output files
                detected_files = self._find_output_files(code)
                logger.info(f"Detected output files: {detected_files}")

                #if "pip" in code:
                #    code = code.replace("!pip", "pip")
                #    code = silence_pip(code, lang)
                #    lang = "bash"
                
                if "pip install" in code:
                    code_lines = code.splitlines()
                    clean_code_lines = []
                    for line in code_lines:
                        if line.strip().startswith("pip install"):
                            dependency_list = line.strip().split()[2:]  # Split and extract dependencies
                            formatted_dependencies = []

                            for dependency in dependency_list:
                                # Preserve the version specifiers by quoting the dependency string
                                if any(op in dependency for op in ['<', '>', '=', '!', '~']):
                                    formatted_dependencies.append(f'"{dependency}"')
                                else:
                                    formatted_dependencies.append(dependency)

                            formatted_line = f"pip install {' '.join(formatted_dependencies)}"
                            clean_code_lines.append(silence_pip(formatted_line, "bash"))
                        else:
                            clean_code_lines.append(line)
                    
                    code = "\n".join(clean_code_lines)
                    lang = "bash"   
            

                if lang not in self.SUPPORTED_LANGUAGES:
                    logger.error(f"Unsupported language: {lang}")
                    return CommandLineCodeResult(exit_code=1, output=f"Unsupported language: {lang}")

                code_hash = md5(code.encode()).hexdigest()
                
                if filename == None:
                    filename = f"tmp_code_{code_hash}.{'py' if lang.startswith('python') else lang}"

                logger.info(f"Processing code block for language: {lang} with filename: {filename}")
                # Write the code to the remote server
                self._write_code_remote(coding_session_id=session_id, filename=filename, code=code)

                # Execute the code on the remote server
                result = self._execute_remote(coding_session_id=session_id, command=f"{_cmd(lang)} {filename}", detach=True)
                job_id = result.get("job_id")
                self.job_list.append(job_id)
                if job_id:
                    logger.info(f"Job {job_id} started, waiting for completion...")
                    # Poll for the job status and logs
                    while True:
                        status = self.client.session_manager.get_job_status(job_id)
                        logger.info(f"Job {job_id} status: {status['status']}")
                        if status['status'] == 'completed':
                            logs_all = ""
                            logs_all += status.get("stdout", "") or ""  # Avoid concatenating None
                            logs_all += status.get("stderr", "") or ""
                            exit_code = status.get("exit_code", 0)

                            if exit_code != 0:
                                error_output = self._parse_errors(logs_all, lang) or ""  # Ensure error_output is always a string
                                if error_output == 'No errors detected. Execution was successful.':
                                    error_output = "Read through the execution log and figure out what went wrong."
                            
                    
                            logger.info(f"Job {job_id} completed with exit code {exit_code}\n detailed_status: {status}")
                            break
                        elif status['status'] == 'running':
                            logs = self._get_job_logs(job_id)
                            stdout_logs = logs.get("stdout", "") or ""  # Ensure no NoneType
                            stderr_logs = logs.get("stderr", "") or ""  # Ensure no NoneType
                            print(stdout_logs, end="")
                            print(stderr_logs, end="")
                            logs_all = stdout_logs + stderr_logs
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
            if False:
                try:
                    self.client.session_manager.close_session(session_id)
                    logger.info(f"Session {session_id} closed successfully.")
                except Exception as e:
                    logger.error(f"Error closing session: {e}")
        trimmed_logs = trim_logs(logs_all)
        return CommandLineCodeResultWithArtifact(exit_code=exit_code, output=trimmed_logs + "\nError Output:" + error_output, code_file=filename, artifacts = saved_files)

    def cleanup(self):
        # self.client.container_manager.terminate_container()
#        self.client.session_manager.close_session(coding_session_id=self.session_info["coding_session_id"])
        for job_id in self.job_list:
            self.client.session_manager.terminate_subprocess(job_id=job_id)

        coding_session_id=self.session_info["coding_session_id"]
        self.client.session_manager.delete_tmp(coding_session_id=coding_session_id)
        self.client.session_manager.close_session(coding_session_id=coding_session_id)
        
# Usage Example: Working
if __name__ == "__main__":
    # Initialize the client with the actual base URL and token
    client = MonsterNeoCodeRuntimeClient(container_type = "gpu")

    try:

        executor = MonsterRemoteCommandLineCodeExecutor(client=client)
        gpu_list = """
#!/bin/bash
# Filename: nvidia_smi.sh
nvidia-smi
"""

        dep_installation = """
#!/bin/bash
pip install scikit-learn numpy matplotlib opencv-python pycuda
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

        gpu_example_code = """
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# CUDA kernel
cuda_code = \"""
__global__ void vector_add(float *a, float *b, float *c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}
\"""

# Compile the CUDA kernel
mod = SourceModule(cuda_code)

# Get the kernel function
vector_add = mod.get_function("vector_add")

# Set up the data
n = 1000000
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)

# Allocate memory on the GPU
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Copy data to the GPU
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Set up the grid and block sizes
block_size = 256
grid_size = (n + block_size - 1) // block_size

# Launch the kernel
vector_add(
    a_gpu,
    b_gpu,
    c_gpu,
    np.int32(n),
    block=(block_size, 1, 1),
    grid=(grid_size, 1)
)

# Copy the result back to the host
cuda.memcpy_dtoh(c, c_gpu)

# Verify the result
np.testing.assert_almost_equal(c, a + b)
print("GPU computation successful!")
"""


        result = executor.execute_code_blocks([CodeBlock(code=dep_installation, language="bash"), CodeBlock(code = gpu_example_code, language="python"), CodeBlock(code=long_running_python_code, language="python")])
        logger.info(f"Final Output:\n{100*'#'}\n{result.output}")
        logger.info(f"Saved Files: {result.artifacts}")

    except Exception as e:
        import traceback;traceback.print_exc()
