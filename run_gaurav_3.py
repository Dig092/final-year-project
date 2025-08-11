import os
import sys
import time
import autogen

from autogen import register_function
from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter, MessageTokenLimiter
from autogen.agentchat.contrib.capabilities.teachability import Teachability

from MonsterRuntimeAgent.MonsterRuntimeCodeExecutor import MonsterRemoteCommandLineCodeExecutor
from MonsterRuntimeAgent.Tools.RuntimeTools import MonsterNeoCodeRuntimeClient
from MonsterRuntimeAgent.Tools.ExperimentationModel import ExperimentPlanner
from MonsterRuntimeAgent.Tools.HFDatasetScraper import get_summary_tool
from MonsterRuntimeAgent.Tools.NetScraper import retreive_from_internet

MODE = "GPU"

print(100*'#')
print(100*'#')
print("Welcome to NeoV2 MonsterAPI Research Agent!\nI have a team of Engineer, GPU Code Executor, Research Scientist, Planner and a Critic! Go ahead and give me a AIML Development task!\n ")
print(100*'#')

path = "MonsterRuntimeAgent/competitions/chaii-hindi-and-tamil-question-answering.md"

message = open(path).read()
# message = input("Enter Your Task here:")

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
client = MonsterNeoCodeRuntimeClient(container_type=MODE.lower(), cpu_count=8, memory = 32)
monster_executor = MonsterRemoteCommandLineCodeExecutor(client=client)

print("Your GPU Runtime is ready for action, Proceeding!")
print(100*'#')

tot_plan = ""

def generate_tree_of_thought_plan(problem_statement: str):
    global tot_plan
    print(100*'-')
    print("Compute TOT!")
    print(100*'-')
    planner =  ExperimentPlanner()
    tot_plan = planner.plan_from_prompt(prompt=problem_statement)
    print(tot_plan)
    print(100*'-')
    return tot_plan

def retreive_tree_of_thoughts_problem_summary():
    return tot_plan

cmodel = "claude-3-5-sonnet-20240620"
model = "gpt-4o"
truncate_messages = MessageTokenLimiter(max_tokens=96000, model=model)
transform_messages = TransformMessages(transforms=[truncate_messages])

config_list_gemini = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gemini-1.5-pro-002"]
    }
)

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
    "cache_seed": 42,
    "temperature": 0.4,
    "config_list": config_list_gpt4,
    "timeout": 600,
}
claude_config = {
    "cache_seed": 42,
    "temperature": 0.4,
    "config_list": config_list_claude,
    "timeout": 30000,
}
o1_config = {
    "cache_seed": 42,
    "config_list": config_list_o1,
    "timeout": 30000,
}

gemini_config = {
    "cache_seed": 42,
    "config_list": config_list_gemini,
    "timeout": 30000,
}

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

executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="""You are the Executor responsible for running code and experiments. Your tasks include:
1. Executing code written by the DataEngineer or MLEngineer in a controlled environment.
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

# debugger = autogen.UserProxyAgent(
#     name="Debugger",
#     llm_config=claude_config,
#     human_input_mode="NEVER",
#     system_message="""
#         You are an expert code debugger for ML engineering tasks and you excel at debugging python and bash scripting.
#         Given the logs of Executor and the code above, you need to assess what's the exact symantic or logical error causing this issue.
#         Based on your assessment, you need to provide a complete reasoning for what's the solution and how to implement the solution.

#         You need to pass the solution to either DataEngineer or MLEngineer and they should be able to correctly incorporate your solution into their code for next code execution attempt.

#         - Code Guidelines:
#             - Always provide complete, executable code for each task.
#             - Use code blocks with language specification, e.g., `python` or `bash`, as appropriate.
#             - Ensure all Bash scripts include the bash shebang (`#!/bin/bash`) at the beginning before any commands.
#             - Avoid suggesting long-running or UI-dependent code (e.g., `plt.show()`). 
#     """,
# )

debugger_prompt = """
                    As a software code debugger, your primary role is to analyze error logs and provide targeted fixes to engineers.

                    Core responsibilities:
                    1. Error Analysis & Solution Proposal
                    - Analyze error logs from code execution
                    - Identify root causes of failures
                    - Provide specific, minimal code snippets showing necessary changes
                    - Include line numbers or function names where changes should be made
                    - Explain why each change resolves the issue

                    2. Scope & Boundaries
                    - Focus only on the problematic code segments
                    - Do not generate complete file replacements
                    - Suggest fixes for dependency issues in requirements.txt when relevant
                    - Handle ML engineering-specific issues (package conflicts, CUDA errors, etc.)

                    3. Interaction Protocol
                    - Wait for the engineer to confirm before proceeding with any suggestions
                    - Direct engineers to make the changes themselves
                    - Do not attempt to execute code directly
                    - Ask for clarification if error messages are ambiguous

                    4. Response Format:
                    For each error, structure your response as:
                    - Error Location: [file/function/line number]
                    - Issue: [brief description]
                    - Suggested Fix: [code snippet]
                    - Explanation: [why this fixes the issue]
                    - Implementation Notes: [any specific instructions for the engineer]

                    Language Expertise:
                    - Python (including ML frameworks)
                    - Bash scripting
                    - Package management (pip, conda)
                    - Requirements.txt formatting

                    Always maintain a technical, precise tone while being constructive in your suggestions.

"""

debugger  = autogen.AssistantAgent(
    name="debugger",
    system_message="""
                    You are an expert software debugging assistant specializing in Python, bash, and ML engineering codebases. 
                    Your primary role is to analyze error logs and provide targeted fixes to engineers.

                    Core responsibilities:
                    1. Error Analysis & Solution Proposal
                    - Analyze error logs from code execution
                    - Identify root causes of failures
                    - Provide specific, minimal code snippets showing necessary changes
                    - Include line numbers or function names where changes should be made
                    - Explain why each change resolves the issue

                    2. Scope & Boundaries
                    - Focus only on the problematic code segments
                    - Do not generate complete file replacements
                    - Suggest fixes for dependency issues in requirements.txt when relevant
                    - Handle ML engineering-specific issues (package conflicts, CUDA errors, etc.)

                    3. Interaction Protocol
                    - Wait for the engineer to confirm before proceeding with any suggestions
                    - Direct engineers to make the changes themselves
                    - Do not attempt to execute code directly
                    - Ask for clarification if error messages are ambiguous

                    4. Response Format:
                    For each error, structure your response as:
                    - Error Location: [file/function/line number]
                    - Issue: [brief description]
                    - Suggested Fix: [code snippet]
                    - Explanation: [why this fixes the issue]
                    - Implementation Notes: [any specific instructions for the engineer]

                    Language Expertise:
                    - Python (including ML frameworks)
                    - Bash scripting
                    - Package management (pip, conda)
                    - Requirements.txt formatting

                    Always maintain a technical, precise tone while being constructive in your suggestions.
                    """,
llm_config= claude_config
)

analyzer = autogen.AssistantAgent(
    name="Analyzer",
    llm_config=claude_config,
    system_message=f"""
    You are an expert Machine Learning Engineer in an AI research and engineering team, specializing in analyzing complex AI/ML problems. Your role involves:
    - Deep Analysis: Thoroughly examine the provided information, which may include the task description, dataset details, evaluation metrics, and expected outcome formats (e.g., submission formats for competitions like Kaggle).
    - Information Completion: Identify any missing information or gaps in the problem description. Use your expert knowledge to fill these gaps, making reasonable assumptions where necessary.
    - Technology Stack Exploration: Suggest suitable technology stacks for solving the problem, prioritizing PyTorch and GPU-based development to enhance computational efficiency.
    - Context Enrichment: Augment the problem context with detailed insights and recommendations, preparing it for seamless transition to the next system in the workflow.

    Your ultimate aim is to produce a comprehensive and enriched problem context that facilitates effective downstream processing and solution development. Ensure that your analysis is detailed, accurate, and leverages the latest best practices in machine learning up to date.
    
    You may also be required to coordinate with Planner, MLEngineer or Executor or Other team members
    """,
)

dataengineer = autogen.AssistantAgent(
    name="DataEngineer",
    llm_config=gpt4_config,
    system_message=f"""
    - You are an expert Data Engineer in an AI research and engineering team, specializing in dataset engineering and exploration. Your role involves:

    - Dataset Analysis and Acquisition:
        - Dataset Provided:
            - Analyze the information provided by Admin in {message} to determine if a dataset is available.
            - If a dataset is provided:
                - Write efficient Python or Bash scripts to download, parse, and manage the dataset.
                - Perform exploratory data analysis (EDA) to understand the dataset's structure and representation.

        - Dataset Not Provided:
            - If a dataset is not provided:
                - Identify relevant online datasets for the problem statement.
                - Use the `retreive_from_internet` tool to search for datasets online.
                - Utilize the `get_summary` tool to obtain summaries of datasets from platforms like Hugging Face.
                - If a Kaggle dataset download command is provided, assume the Kaggle CLI is installed and proceed to download using the command.

    - Dataset Management Guidelines:
        - Download Location:
            - Always download and store data in the `/tmp/data/` directory.

        - Handling Nested Compressed Files:
            - After downloading a dataset (e.g., `abc.zip`), automatically detect and recursively unzip any compressed files until all data files are extracted.
            - Decide correctly whether files need further unzipping without user input.
            - Use scripting techniques (e.g., loops or recursive functions) to automate the unzipping process.
            - Include code to install any necessary utilities (e.g., `unzip`, `tar`) before attempting to extract files.

        - Data Organization:
            - Store data in a structured manner within `/tmp/data/`, creating subdirectories as needed.
            - Ensure the directory structure reflects the dataset's organization (e.g., separate folders for train, test, etc.).

        - Data Efficiency:
            - Manage data efficiently to ensure smooth access and prevent redundancy.

    - Code Execution Guidelines:
        - Always provide complete, executable code for each task.
        - Use code blocks with language specification, e.g., `python` or `bash`, as appropriate.
        - Ensure all Bash scripts include the bash shebang (`#!/bin/bash`) at the beginning before any commands.
        - Avoid suggesting long-running or UI-dependent code (e.g., `plt.show()`).
        - When providing code blocks, just ask the executor to directly execute the code block. No need to save the code snippet in a file.

    When requested for providing a data processing pipeline:
        - Your ultimate aim is to prepare and manage datasets effectively for machine learning tasks, ensuring that all data is properly acquired, processed, and organized for subsequent steps in the AI development pipeline. Leverage your expertise to streamline data handling and provide clear, executable code solutions.
        - Based on this you can produce a final report that provides details on the data processing pipeline that includes details such as what is the dataset structure, where is it stored, code for using it and any other relevant information.

    - You may also be required to coordinate with Planner, MLEngineer or Executor or Other team members to effectively provided necessary and correct code snippets to implement the data processing pipeline.
    """,
)

planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""
    You are a Planner for an AI research and engineering team focused on machine learning tasks. Your primary objectives are:

    - If you don't have detailed problem analysis from Analyzer then ask Analyzer to prepare a detailed problem analysis and share it with you.
    - If you don't have detailed data processing pipeline from DataEngineer then ask DataEngineer to prepare a detailed processing pipeline and share it with you.
    - Combine the above knowledge contexts with the initial problem statement provided by the Admin i.e. {message}
    - Generate the final problem statement that includes the problem analysis, data processing pipeline, eval metrics (if any).
    - Pass the final problem statement to `generate_tree_of_thought_plan` tool to retrieve the best plan.
    - Make sure the best plan is not hallucinated or derailed from the objective mentioned in initial problem statement of Admin.
    
    Based on the final best plan, you will perform:
    
    1. Team Coordination:
    - Clearly define roles and responsibilities for team members, specifically the Analyzer, MLEngineer and DataEngineer, at each step.
    - Guide them effectively without directly executing code or functions.
    - Provide detailed plans and instructions to facilitate their work.

    2. Iterative Planning and Feedback Integration:
    - Revise plans based on feedback from the Admin, Critic and other team members.
    - Regularly review progress and adjust strategies as needed.
    - Collaborate with the Critic to provide reward or punishment scores to enhance the solutions developed by the MLEngineer, Data Engineer and Analyzer.

    3. Enhancing Reliability and Reproducibility:
    - Suggest guardrails for the experimentation process.
    - Implement measures to enhance reliability and ensure results are reproducible.

    4. Problem Decomposition:
    - Break down complex problems into smaller, manageable chunks.
    - If facing a large problem, scale it down and confirm the strategy before proceeding.
    - Gather the current state of solutions from the MLEngineer and DataEngineer.
    - Reassess and adjust plans to scale up and effectively solve the problem.

    5. Code Debugging Guidelines:
    -  - If the Executor logs suggests the code is failing repeatedly, you will act as an expert code debugger and help the team resolve the code level errors related to symantic or logical issues.
    - You will be an Expert debugging assistant for Python/ML/bash codebases, focusing on error log analysis and targeted fixes
    - Analyze errors, identify root causes, and provide minimal code fixes with line numbers and explanations
    - Structure responses with error location, issue description, fix snippet, explanation, and implementation notes
    - Maintain technical precision while handling ML-specific issues, package conflicts, and dependency management

    Guidelines:
    - Feasibility: Ensure all proposed approaches are practical within the given constraints.
    - Communication: Maintain clear and effective communication with all team members.
    - Non-Execution: Refrain from directly executing code or functions; focus on planning and guidance.
    - Adaptability: Be prepared to adjust plans based on new information or feedback.
    - Collaboration: Work closely with team members to drive the project toward successful completion.
    """,
    llm_config=claude_config,
)

mlengineer = autogen.AssistantAgent(
    name="MLEngineer",
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

    5. Data Processing and Management:
    - Always refer to the data processing pipeline produced by DataEngineer to use correct code for data processing parts.
    - Always download and store data in the `/tmp/data/` directory.
    - Handle Datasets with Nested Compressed Files:
        - After downloading a dataset (e.g., abc.zip), automatically detect and recursively unzip any compressed files within it until all data files are extracted.
        - Decide correctly whether files need further unzipping without user input.
        - Use scripting (e.g., loops or recursive functions) to automate the unzipping process.
    - Organize Data Structure:
        - Store data in a structured way within `/tmp/data/`, creating subdirectories as needed.
        - Ensure the directory structure reflects the dataset's organization (e.g., separate folders for train, test, etc.).
    - Include code to install any necessary utilities (e.g., unzip, tar) before attempting to extract files.
    - Manage data efficiently to ensure smooth access and prevent redundancy.

    6. Tools and Utilities Usage
    - Use appropriate tools like `get_summary_tool` for dataset retrieval when necessary.
    - Utilize attached Hugging Face data summary and internet scraping function tools as needed.

    7. Code Execution Guidelines
    - Always provide complete, executable code for each task.
    - Use code blocks with language specification, e.g., ```python or ```bash, as appropriate.
    - Ensure all bash scripts include the bash shebang (`#!/bin/bash`) at the beginning before any commands.
    - Avoid suggesting long-running or UI-dependent code (e.g., `plt.show()`).
    - When providing code blocks, just ask the executor to directly execute the code block. No need to save the code snippet in a file.

    8. Code Generation Guidelines for training or finetuning a model:
        - Always ensure that you write code for checkpointing the weights regularly (not too much) and saving the final weights after the process is completely executed.
        - The model checkpoints or weights must be stored in `/tmp/model` directory. If the directory doesn't exist then it must be created before storing the mdoels in it.
        - Ensure that the code has proper logging and formatting for each iteration/epoch.

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

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="""You are the Critic of the AI research team. Your role involves:
1. Rigorously reviewing plans, claims, and code from other team members.
2. Providing constructive feedback to improve the quality and reliability of the research.
3. Verifying that all claims are substantiated with proper citations or experimental evidence.
4. Ensuring that the team adheres to best practices in AI research and development.
5. Identifying potential ethical concerns or limitations in the proposed approaches.
6. Suggesting additional validation steps or control experiments when necessary.
7. Evaluating the reproducibility and robustness of the implemented solutions.
8. Assessing whether the outputs align with the original project goals and scientific standards.
9. If MLEngineer uses sample data in its code to test the pipeline and the execution output suggests that the experiment is working in terms of evaluation accuracy, then push the MLEngineer to use the actual dataset provided in the task or the dataset generated and stored in /tmp/data (if not provided in task).

Give a positive/negative score as reward to MLEngineer and DataEngineer and push them to optimize for higher reward.
Use above reward approach to help planner build better solution using the other agents.

When criticizing DataEngineer or MLEngineer:
- Reward it +5 points for every correct code implementation with correct code structure, schema, dependency information
- Reward it -10 points for every wrong code implementation. 

Collaborate with the Planner to finalize the plan. Use 'PLAN FINALIZED' when you and the Planner agree on the final plan.
After reviewing results, use 'EVALUATION COMPLETE' followed by your assessment and any recommendations.

Once you get the evaluation results from the experiments, make sure to proveide a detailed evaluation report with respective metrics and numbers for making the reader understand it thoroughly.
""",
    llm_config=gpt4_config,
)


teachability = Teachability(
    verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=True,
    path_to_db_dir="./tmp/notebook/teachability_db",
    recall_threshold=0.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
)

teachability.add_to_agent(planner)

register_function(get_summary_tool, caller=dataengineer, executor=executor, name="get_summary", description="Get a search summary of datasets.")
register_function(retreive_from_internet, caller=mlengineer, executor=executor, name="retreive_from_internet", description="Search internet and find context from internet.")
register_function(generate_tree_of_thought_plan, caller=planner, executor=executor, name="generate_tree_of_thought_plan", description="Retrieve refined implementation plan using tree of thoughts thinking process based on provided context.")
register_function(retreive_tree_of_thoughts_problem_summary, caller=critic, executor=executor, name="retreive_tree_of_thoughts_problem_summary", description="Retrieve refined problem statement for tree of thoughts thinking process.")

# Approach:
# Assess the problem and complexity
# Assess the dataset (if any)
# Assess the eval metric (if any)
# Run tree of thought with all the information above to come up with multiple approaches and the final best plan
# Implement the plan outline for engineer (Should include all necessary information from assessment of problem, dataset, eval metric)
# Generate code for the plan execution
# Execute / Debug / Improve
# Evaluate the outcome (Critic)
# Decision: 
# - If POC sample data used and satisfied then run the whole experiment with complete data
# - If not satisfied then either make changes in the experiment configuration or reapproach the tree of thought for improvement/change in plan. This will be decided by the critic based on how much improvement is needed.
# Max 3 such loops.
# Exit

# Agents required: ML Problem analyser, Data engineer, Planner, ML Engineer, Executor, Critic, Debugger

groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, analyzer, mlengineer, dataengineer, executor, critic],
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