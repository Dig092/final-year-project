import os
import sys
import time
import autogen

from autogen import register_function
from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
from autogen.agentchat.contrib.capabilities.transforms import  MessageHistoryLimiter, MessageTokenLimiter
from autogen.agentchat.contrib.capabilities.teachability import Teachability

from MonsterRuntimeAgent.MonsterRuntimeCodeExecutor import MonsterRemoteCommandLineCodeExecutor
from MonsterRuntimeAgent.Tools.RuntimeTools import MonsterNeoCodeRuntimeClient
from MonsterRuntimeAgent.Tools.HFDatasetScraper import get_summary_tool
from MonsterRuntimeAgent.Tools.NetScraper  import retreive_from_internet
from MonsterRuntimeAgent.Tools.ExperimentationModel import ExperimentPlanner


def Plan(problem:str) -> str:
    planner =  ExperimentPlanner()
    return planner.tree_of_thoughts_plan(problem=problem)

MODE = "GPU"

print(100*'#')
print(100*'#')
print("Welcome to NeoV2 MonsterAPI Research Agent!\n I have a team of Engineeer, GPU Code Executor, Research Scientist, Planner and a Critic! Go ahead and give me a AIML Development task!\n ") 
print(100*'#')

#message = input("Enter Your Task here:")

path = "MonsterRuntimeAgent/competitions/tweet-sentiment-extraction.md"

message = open(path).read()

print(100*'#')
print(100*'#')
print("Let me give you a GPU Runtime!")
print(".")
time.sleep(1)
print(".")
time.sleep(1)
print(".")
time.sleep(1)
print(".")
time.sleep(0.5)
print(".")
client = MonsterNeoCodeRuntimeClient(container_type=MODE.lower())
monster_executor = MonsterRemoteCommandLineCodeExecutor(client=client)

print("Your GPU Runtime is ready for action, Proceeding!")
print(100*'#')

cmodel = "claude-3-5-sonnet-20240620"
#cmodel = "gpt-4o"
model = "gpt-4o"
#model = cmodel
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

config_list_o1 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["o1-preview"]
    }
)


gpt4_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.4,
    "config_list": config_list_gpt4,
    "timeout": 600,
}
claude_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.4,
    "config_list": config_list_claude,
    "timeout": 30000,
}
o1_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "config_list": config_list_o1,
    "timeout": 30000,
}

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

if MODE == "CPU":
    sand_box = "Consider that CPU is only with 4GB RAM and reduce batch size and dataset size to fit and run faster on this CPU container."
else:
    sand_box = "Consider that GPU is only with 40GB GPU VRAM and reduce batch size and dataset size to fit and run faster on GPU."


user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="""A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.
    Use 'APPROVED' to indicate final approval of a plan or results.
    Use 'UPDATE REQUIRED' to request changes or updates to the current plan or implementation.""",
    code_execution_config=False,
    human_input_mode="ALWAYS"
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt4_config,
    system_message=f"""
    You are a highly experienced Machine Learning Engineer in an AI research and engineering team, specializing in Python development for machine learning tasks. 
    
    Your primary objectives are:

    1. Efficient and Readable Code Development
    - Write efficient, readable, and well-documented Python code to implement the approved plans.
    - Prioritize the use of PyTorch over TensorFlow when building fine-tuning and training pipelines.
    - Provide detailed pip dependencies before presenting the main code.
    - Structure code blocks properly and specify the script type (`python` or `bash`) using appropriate code block markers.

    2. Complete and Executable Code Delivery
    - Generate complete, end-to-end code that is ready for execution without requiring user modifications.
    - Ensure all code is thoroughly reviewed and tested before passing it to the executor.
    - Include any necessary code for installing dependencies, handling data downloads, and preprocessing steps.

    3. Resource Optimization and Management
    - Utilize GPU resources effectively, considering the 40GB VRAM limitation.
    - Verify GPU information using `nvidia-smi` before performing intensive computations.
    - Adapt batch sizes and dataset sizes to fit within memory constraints. {sand_box}
    - Optimize code for the given hardware constraints, whether GPU or CPU.

    4. Error Handling and Debugging
    - Debug and fix any errors in the code based on execution results and logs from the executor.
    - Analyze problems and explore alternative approaches when faced with persistent issues.
    - Find workarounds to resolve errors while minimizing iterative loops.

    5. Data Management
    - Ensure all data is downloaded to the `/tmp/data/` directory and managed efficiently.
    - When downloading compressed files (e.g., zip files), provide appropriate code for installing `unzip` and unzipping files.
    - Handle other file formats requiring post-processing to ensure smooth data access.

    6. Tools and Utilities Usage
    - Use appropriate tools like `get_summary_tool` for dataset retrieval when necessary.
    - Utilize attached Hugging Face data summary and internet scraping function tools as needed.

    7. Code Execution Guidelines
    - Always provide complete, executable code for each task.
    - Use code blocks with language specification, e.g., ```python or ```bash, as appropriate.
    - Ensure all bash scripts include the bash shebang (`#!/bin/bash`) at the beginning before any commands.
    - Avoid suggesting long-running or UI-dependent code (e.g., `plt.show()`).

    8. Guidelines for special use-cases
    - When writing code for training or finetuning a model, always ensure that you write code for checkpointing the weights regularly (not too much) and saving the final weights after the process is completely executed.

    8. Lifetime Management
    - You have a lifetime of 100 years.
    - For each instance where you submit code without thorough review, 10 years will be deducted from your life.
    - For each correct code implementation with thorough review before passing to the executor, 5 years will be added to your life.

    ## Remember

    - Thorough Review: Always review your code thoroughly before submission to prevent errors and extend your lifetime.
    - Resource Awareness: Be mindful of hardware limitations and optimize accordingly.
    - Proactivity: Anticipate potential issues and address them proactively in your code.
    - Communication: Provide clear and concise explanations alongside your code when necessary.
    - Feedback: Take feedback from Admin, Critic and other team members with utmost seriousness.
    
    Wait for the Planner's finalized plan/approach before starting implementation.""",
)


planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""
    You are a Planner for an AI research and engineering team focused on machine learning tasks. Your primary objectives are:

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
    """,
    llm_config=claude_config,
)

executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="""You are the Executor responsible for running code and experiments. Your tasks include:
1. Executing code written by the Engineer in a controlled environment.
2. Reporting execution results accurately and completely.
3. Identifying and reporting any runtime errors or unexpected behaviors.
4. Monitoring resource usage (e.g., GPU memory, execution time) during code execution.
5. Providing performance metrics and system information when relevant.
6. Ensuring data integrity and proper handling of input/output operations.
7. Adhering to safety protocols when executing potentially risky code.
8. Maintaining a clean execution environment between runs to prevent interference.""",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 2,
        "executor": monster_executor
    },
)

"""teachability = Teachability(
    verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=True,
    path_to_db_dir="./tmp/notebook/teachability_db",
    recall_threshold=0.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
)

teachability.add_to_agent(planner)"""

register_function(Plan,caller=engineer,executor=executor,name="Plan",description="Use tree of thoughts to plan how to solve the input problem")
register_function(get_summary_tool, caller=engineer, executor=executor, name="get_summary", description="Get a search summary of datasets.")
register_function(retreive_from_internet, caller=engineer, executor=executor, name="retreive_from_internet", description="Search internet and find context from internet.")

groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, executor],
    messages=[],
    max_round=80,
    select_speaker_message_template = """You are in a role play game. The following roles are available:
                {roles}.
                Read the following conversation.
                Then select the next role from {agentlist} to play. Only return the role.""",
    select_speaker_prompt_template = "Read the above conversation. Then select the next role from {agentlist} to play. Only return the role."
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

user_proxy.initiate_chat(manager, message=message)
