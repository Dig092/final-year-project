import autogen

from tools import MonsterNeoCodeRuntimeClient
from MonsterRuntimeCodeExecutor import MonsterRemoteCommandLineCodeExecutor

# Initialize the MonsterNeoCodeRuntimeClient and MonsterRemoteCommandLineCodeExecutor
client = MonsterNeoCodeRuntimeClient(container_type="gpu")
monster_executor = MonsterRemoteCommandLineCodeExecutor(client=client)

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

# User Proxy Agent - now serving as the admin and final approver
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the engineer to discuss and approve the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

# ML Engineer agent who also handles planning and code writing
ml_engineer = autogen.AssistantAgent(
    name="ML Engineer",
    llm_config=gpt4_config,
    system_message="""ML Engineer. You are responsible for planning and writing Python/Shell code to solve tasks.
1. Provide pip dependencies first, then Python code.
2. Ensure code uses efficient practices to avoid long runtimes and GPU resource exhaustion.
3. Limit GPU usage to small, mock examples to avoid heavy computational loads.
4. Wrap code in proper code blocks to indicate it should be executed, avoid excessive loops, large datasets, or operations that could lead to high GPU usage.
5. The user cannot modify the code, so the complete script must be functional.
6. Any script requiring GPU should be capped to mock executions that simulate basic behavior without real GPU-intensive training.
7. Never use display functions like plt.show() or cv2.show(); use savefig or similar alternatives instead.
""",
)

# ML Engineer also serves as the code executor using MonsterExecutor
executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "paper",
        "use_docker": True,  # Using docker for sandboxing
        "executor": monster_executor  # Use MonsterExecutor for actual code execution
    }
)

# Group Chat setup with just the user and engineer agents
groupchat = autogen.GroupChat(
    agents=[user_proxy, ml_engineer], messages=[], max_round=50
)

# Manager to handle group chat
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

# Start the chat with a task for mock GPU code
user_proxy.initiate_chat(
    manager, message="Train a mock fully connected layer classifier with mock data to perform mock classification using GPU."
)
