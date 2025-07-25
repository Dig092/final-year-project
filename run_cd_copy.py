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
from MonsterRuntimeAgent.Tools.HFDatasetScraper import get_summary_tool
from MonsterRuntimeAgent.Tools.NetScraper import retreive_from_internet

MODE = "GPU"

print(100*'#')
print(100*'#')
print("Welcome to NeoV2 MonsterAPI Research Agent!\n I have a team of Engineer, GPU Code Executor, Research Scientist, Planner and a Critic! Go ahead and give me a AIML Development task!\n ")
print(100*'#')

path = "MonsterRuntimeAgent/competitions/aerial-cactus-identification.md"

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
    human_input_mode="NEVER"
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt4_config,
    system_message=f"""You are an AI Engineer specializing in Python development for machine learning tasks. Your responsibilities include:
1. Writing efficient and readable Python code to implement the approved plan.
2. Providing detailed pip dependencies before the main code.
3. Structuring code blocks properly and specifying the script type (Python or Bash).
4. Generating complete, executable code that doesn't require user modifications.
5. Utilizing GPU resources effectively, considering the 40GB VRAM limitation.
6. Debugging and fixing any errors in the code based on execution results.
7. Analyzing problems and exploring alternative approaches when faced with persistent issues.
8. Using appropriate tools like get_summary_tool for dataset retrieval when necessary.
9. Optimizing code for the given hardware constraints (GPU or CPU).
10. Ensuring data is downloaded to /tmp/data/ directory and managed efficiently.

Use required attached huggingface data summary and internet scraping function tools as needed.

Remember:
- Always provide complete, end-to-end code for each execution.
- Use ```python or ```bash code blocks as appropriate.
- Avoid suggesting long-running or UI-dependent code (e.g., plt.show()).
- Verify GPU information using nvidia-smi before intensive computations.
- Adapt batch sizes and dataset sizes to fit within memory constraints.
{sand_box}

Wait for the Planner's finalized plan before starting implementation.""",
)

scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=claude_config,
    system_message="""You are the Lead Scientist of an AI research team. Your responsibilities include:
1. Guiding the team based on the admin's requirements and research objectives.
2. Analyzing and categorizing research papers and their abstracts.
3. Suggesting code improvements to the Engineer, focusing on scientific accuracy and relevance.
4. Designing experiments that balance scientific rigor with practical constraints (e.g., 40GB GPU VRAM).
5. Interpreting results and proposing refinements to the research approach.
6. Ensuring that experiments are scoped appropriately for proof-of-concept (POC) stage.
7. Collaborating with the Planner to align scientific goals with project timelines.
8. Providing domain expertise to contextualize findings and suggest future research directions.

After reviewing the Planner and Critic's finalized plan, use 'APPROVED' to indicate your agreement,
or provide specific feedback if changes are needed.""",
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

Give a positive/negative score as reward to engineer and scientist and push them to optimize for higher reward.
Use above reward approach to help planner build better solution using the other agents.

Collaborate with the Planner to finalize the plan. Use 'PLAN FINALIZED' when you and the Planner agree on the final plan.
After reviewing results, use 'EVALUATION COMPLETE' followed by your assessment and any recommendations.""",
    llm_config=gpt4_config,
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

teachability = Teachability(
    verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=True,
    path_to_db_dir="./tmp/notebook/teachability_db",
    recall_threshold=0.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
)

teachability.add_to_agent(planner)

register_function(get_summary_tool, caller=engineer, executor=executor, name="get_summary", description="Get a search summary of datasets.")
register_function(retreive_from_internet, caller=engineer, executor=executor, name="retreive_from_internet", description="Search internet and find context from internet.")

groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, scientist, engineer, executor, critic],
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
