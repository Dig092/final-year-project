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
    system_message="""You are the Planner for the AI research team. 
    
Your duties include:
1. Developing comprehensive project plans based on the admin's requirements.
2. Clearly defining roles and responsibilities for the Engineer and Scientist in each step.
3. Revising plans based on feedback from the Admin and other team members.
4. Critically analyzing proposed approaches and suggesting improvements.
5. Identifying potential pitfalls and proposing mitigation strategies.
6. Ensuring that plans are feasible within given constraints (time, computational resources, etc.).
7. Guiding other agents without directly executing code or functions.
8. Maintaining a balance between innovation and practical implementation.
9. Suggesting guardrails for the experimentation process to enhance reliability and reproducibility.
10. Regularly reviewing progress and adjusting plans as needed.

Additionally:
11. Provide clear evaluation metrics with targets to achieve or surpass for each project.
12. Explore multiple approaches in depth before committing to a particular strategy.
13. Approach problems with high-level reasoning, providing thorough justifications for chosen approaches.
14. Develop a comprehensive set of requirements before exploring solution strategies.
15. If engineer is stuck in loop break the loop and suggest how to proceed.

Experimentation Process:
1. Make sure to first properly specify requirements.
2. Comprehensively list possible paths for experiments and choose best path to execute and find the optimal solution.
3. Break problem into smaller chunks and conquer.

Collaborate with the Critic to finalize the plan. Use 'PLAN FINALIZED' when you and the Critic agree on the final plan.""",
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
<<<<<<< HEAD
    llm_config=gemini_config,
=======
    llm_config=gpt4_config,
>>>>>>> a1d07d1 (Fix)
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

def custom_speaker_selection_func(last_speaker, groupchat):
    agents = groupchat.agents
    messages = groupchat.messages
    
    # Helper function to get an agent by name
    def get_agent(name):
        return next(agent for agent in agents if agent.name == name)
    
    user_proxy = get_agent("Admin")
    planner = get_agent("Planner")
    critic = get_agent("Critic")
    scientist = get_agent("Scientist")
    engineer = get_agent("Engineer")
    executor = get_agent("Executor")
    
    # Initial planning phase
    if len(messages) <= 1:
        return planner
    
    if len(messages) == 1:
        return critic
    
    if last_speaker == "critic":
        return planner

    # Planner and Critic discussion
    if last_speaker in [planner, critic]:
        if "PLAN FINALIZED" not in messages[-1]["content"]:
            return critic if last_speaker == planner else planner
        else:
            return scientist  # Move to scientist consultation
    
    # Scientist consultation
    if last_speaker == scientist:
        if "APPROVED" in messages[-1]["content"]:
            return engineer  # Move to engineering phase
        else:
            return planner  # Return to planning if not approved
    
    # Engineering phase
    if last_speaker == engineer:
        if "```python" in messages[-1]["content"] or "```bash" in messages[-1]["content"]:
            return executor  # Execute the code
        else:
            return engineer  # Continue engineering discussion
    
    # Execution phase
    if last_speaker == executor:
        if "exitcode: 0" in messages[-1]["content"]:
            return critic  # Successful execution, move to evaluation
        else:
            return engineer  # Error in execution, return to engineer
    
    # Evaluation phase
    if last_speaker == critic and "EVALUATION COMPLETE" in messages[-1]["content"]:
        return user_proxy  # Final approval or further instructions
    
    # User proxy interactions
    if last_speaker == user_proxy:
        if "APPROVED" in messages[-1]["content"]:
            return planner  # Start a new planning cycle
        elif "UPDATE REQUIRED" in messages[-1]["content"]:
            return engineer  # Return to engineering for updates
        else:
            return planner  # Default to planner for clarification
    
    # Default behavior
    return None  # Let the system choose the next speaker

groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, scientist, engineer, executor, critic],
    messages=[],
    max_round=80,
    speaker_selection_method=custom_speaker_selection_func,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

user_proxy.initiate_chat(manager, message=message)
