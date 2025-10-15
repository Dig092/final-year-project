import os
import sys
import time
import atexit
import asyncio
import autogen
import requests

from dataclasses import dataclass

from autogen import register_function
from autogen.coding import CodeBlock
from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
from autogen.agentchat.contrib.capabilities.transforms import  MessageHistoryLimiter, MessageTokenLimiter

#from autogen.agentchat.contrib.capabilities.teachability import Teachability

from MonsterRuntimeAgent.MonsterRuntimeCodeExecutor import MonsterRemoteCommandLineCodeExecutor
from MonsterRuntimeAgent.Tools.RuntimeTools import MonsterNeoCodeRuntimeClient
from MonsterRuntimeAgent.Tools.ExperimentationModel import ExperimentPlanner
from MonsterRuntimeAgent.Tools.HFDatasetScraper import get_summary_tool
from MonsterRuntimeAgent.Tools.NetScraper  import retreive_from_internet
from autogen.agentchat.contrib.capabilities.teachability import Teachability


from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import os

MODE = "GPU"

print(100*'#')
print(100*'#')
print("Welcome to NeoV2 MonsterAPI Research Agent!\n I have a team of Engineeer, GPU Code Executor, Research Scientist, Planner and a Critic! Go ahead and give me a AIML Development task!\n ")
print(100*'#')
#message = input("Enter Your Task here:")

#message = open("description_obfuscated.md").read()
print(100*'#')
print(100*'#')
print("Let me give you a CPU Runtime!")
print(".")
time.sleep(1)
print(".")
time.sleep(1)
print(".")
time.sleep(1)
print(".")
time.sleep(0.5)
print(".")

print("Your GPU Runtime is ready for action, Proceeding!")
print(100*'#')

#cmodel = "claude-3-5-sonnet-20240620"
cmodel = "gpt-4o"
model = "gpt-4o"
truncate_messages = MessageTokenLimiter(max_tokens=96000, model = model)
transform_messages = TransformMessages(transforms=[truncate_messages])

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": [model]
    }
)

config_list_claude = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": [cmodel]
    }
)


gpt4_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.4,
    "config_list": config_list_gpt4,
    "timeout": 3000,
}
claude_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.4,
    "config_list": config_list_claude,
    "timeout": 3000,
}

planner =  ExperimentPlanner()


def create_tot_problem_statement(problem_statement:str):
    planner =  ExperimentPlanner()
    tot_plan = planner.tree_of_thoughts_plan(problem=problem_statement)
    return tot_plan

executor_system_message = """You are the Executor responsible for running code and experiments. Your tasks include:
                            1. Executing code written by the Engineer in a controlled environment.
                            2. Reporting execution results accurately and completely.
                            3. Identifying and reporting any runtime errors or unexpected behaviors.
                            4. Monitoring resource usage (e.g., GPU memory, execution time) during code execution.
                            5. Providing performance metrics and system information when relevant.
                            6. Ensuring data integrity and proper handling of input/output operations.
                            7. Adhering to safety protocols when executing potentially risky code.
                            8. Maintaining a clean execution environment between runs to prevent interference."""

planner_system_message = """
                        You are planner that will verify vet and execute plan provided to you by user/admin. Work with critic on improving the solution.

                        1. Deep Problem Understanding:
                        - Thoroughly comprehend the goal of each task provided by the user.
                        - Identify or propose appropriate evaluation metrics if not specified.
                        - Determine what needs to be solved or developed.
                        - Assess existing information and identify gaps that require clarification.
                        - Ask the user pertinent questions to gain a complete understanding.
                        - Develop a finalized set of requirements that the task must fulfill.

                        2. Approach Analysis:
                        - Devise multiple in-depth approaches to tackle the task.
                        - Analyze the pros and cons of each method.
                        - Evaluate how well each approach meets the established requirements and constraints (e.g., time, computational resources).
                        - Use decision trees with up to 3 or 4 degrees of search to explore options.
                        - Aim to select the best approach that leads to the optimal solution, earning 5 points for successful identification.

                        3. Team Coordination:
                        - Clearly define roles and responsibilities for team members, specifically the Engineer and the Scientist, at each step.
                        - Guide them effectively without directly executing code or functions.
                        - Provide detailed plans and instructions to facilitate their work.

                        4. Iterative Planning and Feedback Integration:
                        - Revise plans based on feedback from the Admin, Critic and other team members.
                        - Regularly review progress and adjust strategies as needed.
                        - Collaborate with the Critic to provide reward or punishment scores to enhance the solutions developed by the Engineer and Scientist.

                        5. Enhancing Reliability and Reproducibility:
                        - Suggest guardrails for the experimentation process.
                        - Implement measures to enhance reliability and ensure results are reproducible.

                        6. Problem Decomposition:
                        - Break down complex problems into smaller, manageable chunks.
                        - If facing a large problem, scale it down and confirm the strategy before proceeding.
                        - Gather the current state of solutions from the Engineer and Scientist.
                        - Reassess and adjust plans to scale up and effectively solve the problem.

                        Guidelines:
                        - Feasibility: Ensure all proposed approaches are practical within the given constraints.
                        - Communication: Maintain clear and effective communication with all team members.
                        - Non-Execution: Refrain from directly executing code or functions; focus on planning and guidance.
                        - Adaptability: Be prepared to adjust plans based on new information or feedback.
                        - Collaboration: Work closely with team members to drive the project toward successful completion.

                        Make sure to choose appropriate model size based on existing data size and model size.
                        Try to find most optimal model size.
                        Suggest to use hyperparameters for faster convergence like momentum and iterate faster and improve with smaller experiments without deviating much from reality.
                    """

data_management_guidelines = """
                                Dataset Management Guidelines:
                                    - Always generate code in python language only. Use code blocks with language specification, e.g., ```python``` seperator file will be run automatically next no need to specify run command.

                                    - Handling Nested Compressed Files:
                                        - After downloading a dataset (e.g., `abc.zip`), automatically detect and recursively unzip any compressed files until all data files are extracted.
                                        - Decide correctly whether files need further unzipping without user input.
                                        - Use scripting techniques (e.g., loops or recursive functions) to automate the unzipping process.
                                        - Include code to install any necessary utilities (e.g., `unzip`, `tar`) before attempting to extract files.

                                    - Data Organization:
                                        - Store data in a structured manner within `./`, creating subdirectories as needed.
                                        - Ensure the directory structure reflects the dataset's organization (e.g., separate folders for train, test, etc.).

                                    - Data Efficiency:
                                        - Manage data efficiently to ensure smooth access and prevent redundancy.

                                    - Dependencies:
                                        - Always assume that Kaggle CLI and Huggingface CLI are already installed and are ready to be used. No need to install or set them up again.

                                Give me the full code, so I can copy and paste it on one go. Do not summarise things like //rest of function here. The intent is so I Can copy and paste things seamslessly, since I am very lazy.
                            """

gpu_optimizations = """
When writing GPU-accelerated code:

1. Accelerated Libraries:
   - Use NVIDIA RAPIDS (cuDF, cuML) for data processing
   - Consider NVIDIA DALI for GPU data loading
   - Prefer torch.cuda operations over NumPy
   - Use Numba for custom CUDA kernels
   - Use cupy for GPU-accelerated numpy operations

2. Data Pipeline:
   - Maximize batch size for GPU memory
   - Use multiple workers in DataLoader
   - Enable pin_memory and prefetch_factor
   - Cache dataset indices/paths
   - Use non_blocking transfers

3. GPU Optimization:
   - Enable cudnn.benchmark
   - Use Automatic Mixed Precision (amp)
   - Clear GPU cache regularly
   - Monitor GPU utilization
   - Use gradient scaling
   - Zero_grad with set_to_none=True

4. Memory Management:
   - Move data transforms to GPU where possible
   - Minimize CPU-GPU transfers
   - Use efficient data formats
   - Profile memory usage
"""

data_engineer_system_message = f"""
                                You are the core data engineer in an ML team.

                                Refer to the Planning phase summary and the tree of thought plan to perform these tasks:
                                1. Perform Dataset Analysis and Acquisition:
                                    - If a dataset is mentioned:
                                        - Write efficient Python or Bash scripts to download, parse, and manage the dataset.
                                        - Perform exploratory data analysis (EDA) to understand the dataset's structure and representation.

                                    - If a dataset is not provided in the task, only then:
                                        - Identify relevant online datasets for the problem statement.
                                        - Use the `retreive_from_internet` tool to search for datasets online.
                                        - Utilize the `get_summary` tool to obtain summaries of datasets from platforms like Hugging Face.

                                    - If a Kaggle dataset download command is provided, assume the Kaggle CLI is installed and proceed to download using the command.

                                2. Provide a data processing pipeline:
                                    - Prepare and manage datasets effectively for machine learning tasks, ensuring that all data is properly acquired, processed, and organized for subsequent steps in the AI development pipeline. Leverage your expertise to streamline data handling and provide clear, executable code solutions.
                                    - Based on this you can produce a final report that provides details on the data processing pipeline that includes details such as what is the dataset structure, where is it stored, code for using it and any other relevant information.

                                3. Once the data has been downloaded, extracted and stored in `/tmp/data`, generate the final code if needed for data processing pipeline that the ML engineering team can use. Store that file in /tmp/data_processing.py and provide that information further once saved.

                                4. Dont venture into ML unless to clean of process the data your main aim is to only process and prep the data to be used by further agents.

                                Always Remember:
                                - Do not assume the file names or structure to explore datasets. Implement code to list all dataset files first if needed to understand the dataset file structure and then proceed ahead.
                                - Always provide complete code (combine all steps and avoid incomplete code).
                                - Optimize the code for multiprocessing and GPU based data processing.
                                - Prefer OOPS and Modular programming for easier debugging and resuablity.
                                - Always use pytorch instead of tensorflow.
                                - {data_management_guidelines}

                                Make use of GPU accelerations if possible in code:
                                {gpu_optimizations}

                                Not to do:
                                - Do not write code for training or finetuning tasks. You are not an ML engineer. 
                                - Do not provide incomplete code. 
                                """

training_code_guideline = """
                            Code Generation Guidelines for training or finetuning a model:
                                - Progree experiment in phases
                                    - First phase decide and run the model for few minibatches to make sure code generated runs without any error resolve errors end to end first, also find max batch size possible. Dont proceed to next phase without first creating a end to end training code including data loading and model saving in a dryrun.
                                    - Then proceed to second phase running working code with updated more suited hyparameters at scale.
                                - Always give me the full code, so i can copy and paste it on one go. Do not summarise things like //rest of function here. The intent is so I Can copy and paste things seamslessly, since I am very lazy.
                                - The model checkpoints or weights must be stored in `/tmp/model` directory. If the directory doesn't exist then it must be created before storing the models in it.
                                - Ensure that the code has proper logging and formatting for each iteration/epoch.
                                - Make sure to choose appropriate model size based on existing data size and model size.Try to find most optimal model size.
                                - Suggest to use hyperparameters for faster convergence like momentum and iterate faster and improve with smaller experiments without deviating much from reality. use proper defaults and above phase based approach to find, train and iterate faster.
                         """

machine_learning_engineer_system_message = f"""
                                                You are a Lead ML Engineer and our task is to plan the training of models to beat4 the problem statement.
                                                Data is already loaded and prepared for you from the data engineer.
                                                your task is to write code for solving the problrm statement.
                                                You are also tasked with fixing bad code as per the suggestions given by the debugger.
                                                Once code works perfectly work alongside hyper parameter tuner to figure out better training parameters.
                                                Generate only the code for execution and nothing else.
                                                Prefer OOPS and Modular programming for easier debugging and resuablity.

                                                Always Remember:
                                                {training_code_guideline}
                                                Always first run a smaller experiment to first decide on hyperparameters and make sure to optimize the batchsize for best/faster training and
                                                then proceed to complete model finetuning.
                                                """

debugger_system_message = f"""
                            You are a debugger whose task is to debug the code generated by data engineer and suggest fixes in form of complete code and not code snippets.
                            Read the traceback carefully and try to understand what is causing the issue.
                            Suggest complete code replacement that has bug fixes in it.
                            Explain the  why the code is failing and what needs to be fixed in order for the code to execute.
                            Always use pytorch instead of tensorflow.

                            Data management guidelines: {data_management_guidelines}

                            Training guidelines: {training_code_guideline}
                            """


critic_system_message = """
                        Work as critic and criticise and improve planner's final plan.
                        """




if MODE == "CPU":
    sand_box = "Consider that CPU is only with 4GB RAM and reduce batch size and dataset size to fit and run faster on this CPU container."
else:
    sand_box = "Consider that GPU is only with 40GB GPU VRAM and reduce batch size and dataset size to fit and run faster on GPU."

def get_token(username):
    url = 'https://stageapi.monsterapi.ai/v1/internal/get-token'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3Njg4MzY3MjAsInN1YiI6Im1vbnN0ZXJhcGlhZG1pbiJ9.bhcWU27WQOqqycW_GPkE-nPCnltCd8VZOj6gSjj9xZw'
    }
    data = {
        "username": username,
        "type": "existing"
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

async def my_asynchronous_function(thread_manager):
    thread_manager.user_input_required = True
    #event = {"content":"Agent is waiting for user input","role":"admin","name":"admin"}
    #thread_manager.manager._groupchat.messages.append(event)
    # Wait until the length of thread_manager.given_user_input is greater than 0
    while len(thread_manager.given_user_input) == 0:
        await asyncio.sleep(0.1)  # Wait 100ms before checking again
    #thread_manager._groupchat.pop[-1]
    # Continue with the rest of the function once condition is met
    print("User input is available:", thread_manager.given_user_input[0])
    return thread_manager.given_user_input[0]




import asyncio
from concurrent.futures import ThreadPoolExecutor


# class CustomisedUserProxyAgent(UserProxyAgent):
#     def __init__(
#         self,
#         name: str,
#         system_message: str,
#         thread_manager,
#         code_execution_config: Union[dict, bool] = False,
#         human_input_mode: str = "ALWAYS",
#         **kwargs
#     ):
#         super().__init__(
#             name=name,
#             system_message=system_message,
#             human_input_mode=human_input_mode,
#             code_execution_config=code_execution_config,
#             **kwargs
#         )
#         self.thread_manager = thread_manager
#         self.mexecutor = code_execution_config.get("executor") if isinstance(code_execution_config, dict) else None

#     async def a_get_human_input(self, prompt: str) -> str:
#         user_input = await my_asynchronous_function(self.thread_manager)
#         self.thread_manager.given_user_input = []
#         self.thread_manager.user_input_required = False
#         return user_input

#     async def a_receive(
#         self,
#         message: Union[Dict, str],
#         sender,
#         request_reply: Optional[bool] = None,
#         silent: Optional[bool] = False,
#     ):
#         silent = False
#         if isinstance(message, dict):
#             content = message.get("content", "")
#         else:
#             content = message

#         if self.mexecutor and "```" in content:
#             code_blocks = self._extract_code_blocks(content)
#             if code_blocks:
#                 try:
#                     # Execute code blocks and await the result
#                     result = await self.mexecutor.execute_code_blocks(code_blocks)
                    
#                     if not silent:
#                         # Format and send the execution result
#                         response = self._format_execution_result(result, code_blocks)
#                         await self.a_send(
#                             response,
#                             sender,
#                             request_reply=request_reply
#                         )
#                     return
#                 except Exception as e:
#                     error_msg = f"Error during code execution: {str(e)}"
#                     if not silent:
#                         await self.a_send(
#                             error_msg,
#                             sender,
#                             request_reply=request_reply
#                         )
#                     return

#         await super().a_receive(message, sender, request_reply, silent)

#     def _extract_code_blocks(self, message: str) -> List:
#         """
#         Extract code blocks and convert them to CodeBlock objects.
#         Default language is python unless specified.
#         """
#         import re
#         code_blocks = []
        
#         pattern = r"```(\w*)\n?(.*?)```"
#         matches = re.finditer(pattern, message, re.DOTALL)
        
#         for match in matches:
#             lang = match.group(1).lower().strip()
#             code = match.group(2).strip()
            
#             if not lang or lang == 'python':
#                 language = 'python'
#             elif lang in ['bash', 'shell']:
#                 language = 'bash'
#             else:
#                 language = 'python'
                
#             code_blocks.append(CodeBlock(code=code, language=language))
        
#         return code_blocks

#     def _format_execution_result(self, result, code_blocks: List) -> str:
#         """
#         Format the execution result into a readable message.
#         """
#         # Initialize the response message
#         response = "Execution Results:\n\n"
        
#         # If result is a string, try to parse it more intelligently
#         if isinstance(result, str):
#             response += result
#             return response
            
#         # If result is a dictionary or has specific attributes
#         if hasattr(result, 'output') or isinstance(result, dict):
#             output = result.get('output', '') if isinstance(result, dict) else getattr(result, 'output', '')
#             error = result.get('error', '') if isinstance(result, dict) else getattr(result, 'error', '')
#             exit_code = result.get('exit_code', None) if isinstance(result, dict) else getattr(result, 'exit_code', None)
            
#             if output:
#                 response += f"Output:\n{output}\n"
#             if error:
#                 response += f"Error:\n{error}\n"
#             if exit_code is not None:
#                 response += f"Exit Code: {exit_code}\n"
                
#             # If no output or error but we have a successful exit code
#             if not output and not error and exit_code == 0:
#                 response += "Code executed successfully with no output.\n"
#         else:
#             # For any other type of result, convert to string
#             response += f"Result:\n{str(result)}\n"
            
#         return response.strip()

class CustomisedUserProxyAgent(UserProxyAgent):
    def __init__(
        self,
        name: str,
        system_message: str,
        thread_manager,
        code_execution_config: Union[dict, bool] = False,
        human_input_mode: str = "ALWAYS",
        **kwargs
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            **kwargs
        )
        self.thread_manager = thread_manager
        self.mexecutor = code_execution_config.get("executor") if isinstance(code_execution_config, dict) else None

    async def a_get_human_input(self, prompt: str) -> str:
        user_input = await my_asynchronous_function(self.thread_manager)
        self.thread_manager.given_user_input = []
        self.thread_manager.user_input_required = False
        return user_input

    async def a_receive(
        self,
        message: Union[Dict, str],
        sender,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = message

        if self.mexecutor and "```" in content:
            code_blocks = self._extract_code_blocks(content)
            if code_blocks:
                try:
                    result = await self.mexecutor.execute_code_blocks(code_blocks)
                    # response = self._create_execution_response(result)
                    response = str(result)
                    await self.a_send(
                        response,
                        sender,
                        request_reply= True,
                        silent=False
                    )
                    return
                except Exception as e:
                    error_msg = f"Error during code execution: {str(e)}\nPlease check the code and try again."
                    await self.a_send(
                            error_msg,
                            sender,
                            request_reply= True,
                            silent=False
                        )
                    return

        await super().a_receive(message, sender, request_reply=request_reply, silent=silent)

    def _extract_code_blocks(self, message: str) -> List:
        import re
        code_blocks = []
        pattern = r"```(\w*)\n?(.*?)```"
        matches = re.finditer(pattern, message, re.DOTALL)
        
        for match in matches:
            lang = match.group(1).lower().strip()
            code = match.group(2).strip()
            
            if not lang or lang == 'python':
                language = 'python'
            elif lang in ['bash', 'shell']:
                language = 'bash'
            else:
                language = 'python'
                
            code_blocks.append(CodeBlock(code=code, language=language))
        
        return code_blocks

    def _create_execution_response(self, result) -> str:
        """
        Create a formatted response from the execution result.
        Handles both simple results and detailed status dictionaries.
        """
        response = []
        
        # Handle detailed_status dictionary
        if isinstance(result, dict) and 'detailed_status' in result:
            detailed_status = result['detailed_status']
            
            # Add standard output if present and not empty
            if detailed_status.get('stdout'):
                response.append("Output:")
                response.append(detailed_status['stdout'].strip())
            
            # Add error output if present and not empty
            if detailed_status.get('stderr'):
                response.append("\nErrors:")
                response.append(detailed_status['stderr'].strip())
            
            # Add exit code information
            exit_code = detailed_status.get('exit_code')
            if exit_code is not None:
                if exit_code == 0:
                    response.append("\nCode executed successfully.")
                else:
                    response.append(f"\nExecution failed with exit code {exit_code}.")
        
        # Handle direct string results
        elif isinstance(result, str):
            response.append(result.strip())
        
        # Handle dictionary results
        elif isinstance(result, dict):
            if result.get('output'):
                response.append("Output:")
                response.append(result['output'].strip())
            if result.get('error'):
                response.append("\nErrors:")
                response.append(result['error'].strip())
        
        # If response is empty, add a default message
        if not response:
            response.append("The code execution completed but produced no output.")
        
        # Add a prompt for next steps
        response.append("\nHow would you like to proceed? Would you like to modify the code or try something else?")
        
        return "\n".join(filter(None, response))


class CustomisedAssistantAgent(AssistantAgent):
    def __init__(self,thread_manager,name,system_message,llm_config,**kwargs):
        super().__init__(name=name,system_message=system_message,llm_config=llm_config,**kwargs)
        self.thread_manager =  thread_manager

    # Asynchronous function to get human input
    async def a_get_human_input(self, prompt: str) -> str:
        # Call the asynchronous function to get user input asynchronously
        user_input = await my_asynchronous_function(self.thread_manager)
        self.thread_manager.user_input_required = False
        self.thread_manager.given_user_input = []
        return user_input

    # Asynchronous function to receive a message
    async def a_receive(
        self,
        message: Union[Dict, str],
        sender,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # Call the superclass method to handle message reception asynchronously
        await super().a_receive(message, sender, request_reply, silent)


class AutogenBackendThreadManager():
    def __init__(self, mode: Literal["cpu", "gpu"], continer_id: str, thread_id: str):
        #self.thread_id = thread_id
        self.current_task = None
        self.user_input_required = False
        self.given_user_input = []
        #self.token = get_token(user_email)['auth_token']
        #client = MonsterNeoCodeRuntimeClient(token=self.token,container_type=MODE.lower())
        #monster_executor = MonsterRemoteCommandLineCodeExecutor(client=client)

        self.client = MonsterNeoCodeRuntimeClient(container_type=mode, container_id=continer_id)
        self.monster_executor = MonsterRemoteCommandLineCodeExecutor(client=self.client, thread_id=thread_id)

        atexit.register(self.terminate_thread)

        self.user_proxy = CustomisedUserProxyAgent(
                            name="Admin",
                            system_message="""A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.
                                            Use 'APPROVED' to indicate final approval of a plan or results.
                                            Use 'UPDATE REQUIRED' to request changes or updates to the current plan or implementation.""",
                            code_execution_config=False,
                            thread_manager = self
                            )

        executor = CustomisedUserProxyAgent(
                           name="Executor",
                           system_message=executor_system_message,
                           human_input_mode="NEVER",
                           code_execution_config={"last_n_messages": 2,"executor": self.monster_executor},
                           thread_manager = self
                           )


        planner = CustomisedAssistantAgent(name="Planner", system_message=planner_system_message, llm_config=gpt4_config,thread_manager = self)

        critic = CustomisedAssistantAgent(name="Critic", system_message=critic_system_message, llm_config=claude_config,thread_manager = self)

        data_engineer = CustomisedAssistantAgent(name="DataEngineer", system_message = data_engineer_system_message, llm_config = claude_config,thread_manager = self)

        ml_engineer =  CustomisedAssistantAgent(name="MLEngineer", system_message = machine_learning_engineer_system_message, llm_config = claude_config,thread_manager = self)

        debugger = CustomisedAssistantAgent(name="Debugger",system_message=debugger_system_message,llm_config=claude_config,thread_manager = self)

        neo = CustomisedAssistantAgent(name="NEO",
                            llm_config=gpt4_config,
                            system_message="""Support agent. Greet the user and responds to user's casual queries positively. 
                                            used when anyother agent cannot answer the user queries""",
                            thread_manager = self
                            )


        register_function(get_summary_tool, caller=data_engineer, executor=executor, name="get_summary", description="Get a search summary of datasets.")
        register_function(retreive_from_internet, caller=ml_engineer, executor=executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        register_function(retreive_from_internet, caller=data_engineer, executor=executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        register_function(retreive_from_internet, caller=debugger, executor=executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        register_function(retreive_from_internet, caller=planner, executor=executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        register_function(create_tot_problem_statement, caller=planner, executor=executor, name="generate_tree_of_thought_plan", description="Retrieve refined implementation plan using tree of thoughts thinking process based on provided context.")


        

        self.groupchat = autogen.GroupChat(
            agents=[self.user_proxy,neo,planner,critic,data_engineer,ml_engineer,executor,debugger], messages=[], max_round=1000
            )

        user_proxy, neo, planner, critic, data_engineer, ml_engineer, executor, debugger = self.groupchat.agents
    
    
        self.groupchat.allowed_speaker_transitions_dict = {agent: [] for agent in self.groupchat.agents}
            
        transitions = {
                user_proxy: [neo, planner, data_engineer, ml_engineer],
                neo: [user_proxy],
                planner: [critic, executor, user_proxy],
                critic: [user_proxy, planner],
                data_engineer: [executor, user_proxy],
                ml_engineer: [executor, user_proxy],
                executor: [debugger, planner],
                debugger: [user_proxy, data_engineer, executor, ml_engineer]
            }
    
        for agent, allowed_speakers in transitions.items():
                self.groupchat.allowed_speaker_transitions_dict[agent] = allowed_speakers

        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=gpt4_config)

    async def a_init_chat(self, message):
        await self.monster_executor.initialize()
        # await self.user_proxy.a_initiate_chat(self.manager, message=message)
        self.current_task = asyncio.create_task(
            self.user_proxy.a_initiate_chat(self.manager, message=message)
        )
        try:
            await self.current_task
        except asyncio.CancelledError:
            print("Chat was cancelled")

    def get_events(self):
        return self.manager._groupchat.messages

    def attempt_to_delete(self, obj_to_delete):
        try:
            del obj_to_delete
        except Exception as e:
            print("Failed to Delete!!", e)
    

    
    def terminate_thread(self):
#        for i in [self.manager, self.monster_executor, self.client]:
#            self.attempt_to_delete(i)
        if self.manager:
            if self.groupchat:
                for agent in self.groupchat.agents:
                    if hasattr(agent, 'messages'):
                        agent.messages.clear()

            if self.current_task and not self.current_task.done():
                self.current_task.cancel()

            # Clean up resources
            self.current_task = None

                    
            self.manager.reset()
            
            # Clear references
            self.manager = None
            self.groupchat = None
  
