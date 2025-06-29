import autogen
import os

# Configure GPT-4 model
config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4"]
    }
)

# Update LLM configuration to avoid long-running tasks
gpt4_config = {
    "cache_seed": 42,  
    "temperature": 0,
    "config_list": config_list_gpt4,
    "timeout": 120,  # 2 minute timeout to prevent long executions
}

# User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

# Engineer agent with restrictions on GPU usage and execution complexity
engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt4_config,
    system_message="""Engineer. You follow an approved plan. You write Python/Shell code to solve tasks.
1. Provide pip dependencies first, then Python code.
2. Ensure code uses efficient practices to avoid long runtimes and GPU resource exhaustion.
3. Limit mock GPU usage to small examples to avoid heavy computational loads.
4. Wrap code in proper code blocks to indicate it should be executed, avoid excessive loops, large datasets, or operations that could lead to high GPU usage.
5. The user cannot modify the code, so the complete script must be functional.
6. Any script that requires GPU should be capped to mock executions that simulate basic behavior without real GPU-intensive training.""",
)

# Scientist agent - no change necessary
scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=gpt4_config,
    system_message="Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.",
)

# Planner agent
planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
Ensure the plan is efficient and doesn't involve code that may run too long or utilize excessive resources.
""",
    llm_config=gpt4_config,
)

# Executor agent with Docker and additional execution guardrails
executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "paper",
        "use_docker": True,  # Using docker for sandboxing
        "execution_timeout": 120,  # 2 minute limit for code execution to prevent long runtimes
    }
)

# Critic agent to ensure plan/code validity and prevent excessive usage
critic = autogen.AssistantAgent(
    name="Critic",
    system_message="""Critic. Double check plan, claims, and code from other agents and provide feedback.
Ensure that the code is optimized, doesn't involve long-running processes, and is efficient.
Check that any GPU-related tasks are mock simulations to prevent resource overuse.""",
    llm_config=gpt4_config,
)

# Group Chat setup
groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, scientist, planner, executor, critic], messages=[], max_round=50
)

# Manager to handle group chat
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

# Start the chat with a task for mock GPU code
user_proxy.initiate_chat(
    manager, message="Train a mock fully connected layer classifier with mock data to perform mock classification using GPU simulation."
)
