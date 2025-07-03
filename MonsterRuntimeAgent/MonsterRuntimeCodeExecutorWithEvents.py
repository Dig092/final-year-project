import logging
import os
import re
import time
import atexit
import requests
from hashlib import md5
from typing import List, Union, Dict, ClassVar, Optional, Generator
from queue import Queue
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
        "matplotlib": r"plt\.savefig\(['\"](.*?)['\"]",
        "opencv": r"cv2\.imwrite\(['\"](.*?)['\"]",
        "numpy": r"np\.save\(['\"](.*?)['\"]",
    }

    def __init__(self, client: MonsterNeoCodeRuntimeClient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self.event_queue = Queue()
        atexit.register(self.cleanup)

    def _add_event(self, event: str):
        """Add an event to the queue."""
        self.event_queue.put(event)

    def _detect_language(self, code: str) -> str:
        if re.search(r'^\s*import\s|\s*def\s|\s*class\s|\s*print\(', code, re.MULTILINE):
            return "python"
        if re.search(r'^\s*#!/bin/bash\s|^\s*echo\s|^\s*ls\s|^\s*cd\s', code, re.MULTILINE):
            return "bash"
        if code.startswith("#!"):
            if "python" in code.lower():
                return "python"
            elif "bash" in code.lower():
                return "bash"
        logger.warning("Unable to detect language, defaulting to python.")
        return "python"

    def _parse_errors(self, logs: str, language: str) -> str:
        errors = []
        if language == "python":
            error_patterns = [r'Traceback \(most recent call last\):', r'Error:', r'Exception:']
            for pattern in error_patterns:
                match = re.search(pattern, logs)
                if match:
                    errors.append(logs[match.start():])
                    break
        elif language == "bash":
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
        logger.info(f"Writing code to remote server with filename: {filename}")
        response = self.client.session_manager.write_code(coding_session_id=coding_session_id, filename=filename, code=code)
        logger.info(f"Write Code Response: {response}")

    def _execute_remote(self, coding_session_id: str, command: str, detach: bool = False, workdir: str = None) -> Dict[str, Union[str, int]]:
        logger.info(f"Executing command on remote server: {command} with detach={detach}")
        result = self.client.session_manager.run_subprocess(coding_session_id=coding_session_id, command=command, detach=detach, workdir=workdir)
        return result

    def _get_job_logs(self, job_id: str) -> Dict[str, str]:
        logger.info(f"Fetching logs for job_id: {job_id}")
        return self.client.session_manager.get_job_logs(job_id=job_id)

    def _find_output_files(self, code: str) -> List[str]:
        found_files = []
        for library, pattern in self.COMMON_SAVE_PATTERNS.items():
            matches = re.findall(pattern, code)
            if matches:
                logger.info(f"Detected {len(matches)} output files from {library} in code: {matches}")
                found_files.extend(matches)
        return found_files

    def _retrieve_output_files(self, session_id: str, output_files: List[str]) -> List[str]:
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

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> Generator[str, None, CommandLineCodeResultWithArtifact]:
        logs_all = ""
        exit_code = 0
        filename = None
        session_id = None
        error_output = ""
        saved_files = []

        try:
            retries = 3
            while retries:
                try:
                    session_info = self.client.session_manager.create_session()
                    session_id = session_info["coding_session_id"]
                    self._add_event(f"Session created with ID: {session_id}")
                    yield f"Session created with ID: {session_id}"
                    break
                except (ConnectionError, Timeout) as e:
                    retries -= 1
                    self._add_event(f"Error creating session: {e}. Retrying {retries} more times...")
                    yield f"Error creating session: {e}. Retrying {retries} more times..."
                    time.sleep(2)
                    if not retries:
                        raise

            for code_block in code_blocks:
                lang = self._detect_language(code_block.code)
                code = code_block.code

                detected_files = self._find_output_files(code)
                self._add_event(f"Detected output files: {detected_files}")
                yield f"Detected output files: {detected_files}"

                if "pip" in code:
                    code = code.replace("!pip", "pip")
                    code = silence_pip(code, lang)
                    lang = "bash"

                if lang not in self.SUPPORTED_LANGUAGES:
                    self._add_event(f"Unsupported language: {lang}")
                    yield f"Unsupported language: {lang}"
                    return CommandLineCodeResult(exit_code=1, output=f"Unsupported language: {lang}")

                code_hash = md5(code.encode()).hexdigest()
                filename = f"tmp_code_{code_hash}.{'py' if lang.startswith('python') else lang}"

                self._add_event(f"Processing code block for language: {lang} with filename: {filename}")
                yield f"Processing code block for language: {lang} with filename: {filename}"

                self._write_code_remote(coding_session_id=session_id, filename=filename, code=code)

                result = self._execute_remote(coding_session_id=session_id, command=f"{_cmd(lang)} {filename}", detach=True)
                job_id = result.get("job_id")

                if job_id:
                    self._add_event(f"Job {job_id} started, waiting for completion...")
                    yield f"Job {job_id} started, waiting for completion..."

                    while True:
                        status = self.client.session_manager.get_job_status(job_id)
                        self._add_event(f"Job {job_id} status: {status['status']}")
                        yield f"Job {job_id} status: {status['status']}"

                        if status['status'] == 'completed':
                            logs_all += status.get("stdout", "") or ""
                            logs_all += status.get("stderr", "") or ""
                            exit_code = status.get("exit_code", 0)

                            if exit_code != 0:
                                error_output = self._parse_errors(logs_all, lang) or ""
                                if error_output == 'No errors detected. Execution was successful.':
                                    error_output = "Most Probably is indent error! Please fix indent"

                            self._add_event(f"Job {job_id} completed with exit code {exit_code}")
                            yield f"Job {job_id} completed with exit code {exit_code}"
                            break
                        elif status['status'] == 'running':
                            logs = self._get_job_logs(job_id)
                            stdout_logs = logs.get("stdout", "") or ""
                            stderr_logs = logs.get("stderr", "") or ""
                            self._add_event(stdout_logs)
                            self._add_event(stderr_logs)
                            yield stdout_logs
                            yield stderr_logs
                            logs_all += stdout_logs + stderr_logs
                        else:
                            logs_all += "Unexpected status received.\n"
                            self._add_event(f"Job {job_id} encountered unexpected status: {status['status']}")
                            yield f"Job {job_id} encountered unexpected status: {status['status']}"
                            exit_code = 1
                            break
                        time.sleep(2)
                else:
                    logs_all += "Failed to start detached process.\n"
                    self._add_event("Failed to start detached process.")
                    yield "Failed to start detached process."
                    exit_code = 1
                    break

                if detected_files:
                    saved_files += self._retrieve_output_files(session_id, detected_files)
                    self._add_event(f"Retrieved files: {saved_files}")
                    yield f"Retrieved files: {saved_files}"

        except Exception as e:
            self._add_event(f"Error during code execution: {e}")
            yield f"Error during code execution: {e}"
            logs_all += f"Error: {str(e)}\n"
            exit_code = 1

        finally:
            if session_id:
                try:
                    self.client.session_manager.close_session(session_id)
                    self._add_event(f"Session {session_id} closed successfully.")
                    yield f"Session {session_id} closed successfully."
                except Exception as e:
                    self._add_event(f"Error closing session: {e}")
                    yield f"Error closing session: {e}"

        return CommandLineCodeResultWithArtifact(exit_code=exit_code, output=logs_all + "\n" + error_output, code_file=filename, artifacts=saved_files)

    def cleanup(self):
        self.client.container_manager.terminate_container()

# Usage example
if __name__ == "__main__":
    # Initialize the client with the actual base URL and token
    client = MonsterNeoCodeRuntimeClient(container_type="gpu")

    try:
        executor = MonsterRemoteCommandLineCodeExecutor(client=client)

        dep_installation = """
        #!/bin/bash
        pip install scikit-learn numpy matplotlib opencv-python pycuda
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

        code_blocks = [
            CodeBlock(code=dep_installation, language="bash"),
            CodeBlock(code=gpu_example_code, language="python"),
            CodeBlock(code=long_running_python_code, language="python")
        ]

        # Execute the code blocks and stream the events
        for event in executor.execute_code_blocks(code_blocks):
            print(event)  # This will print each event as it occurs

        # The final result is returned after all events have been yielded
        final_result = executor.execute_code_blocks(code_blocks).__next__()
        print(f"Final Output:\n{100*'#'}\n{final_result.output}")
        print(f"Saved Files: {final_result.artifacts}")

    except Exception as e:
        import traceback
        traceback.print_exc()