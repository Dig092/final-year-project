import os
import sys
import time
import autogen
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

sys.path.append("/home/dev/MDockerRuntimeAPI/MonsterRuntimeAgent/")

from tools import MonsterNeoCodeRuntimeClient
from MonsterRuntimeCodeExecutor import MonsterRemoteCommandLineCodeExecutor

print(100*'#')
print("Welcome to NeoV2 MonsterAPI Research Agent with RAG capabilities!")
print("I have a team of Engineer, GPU Code Executor, Research Scientist, Planner, Critic, and a RAG Assistant!")
print(100*'#')

message = input("Enter Your Task here: ")

print(100*'#')
print("Setting up GPU Runtime and RAG Assistant...")
client = MonsterNeoCodeRuntimeClient(container_type="gpu")
monster_executor = MonsterRemoteCommandLineCodeExecutor(client=client)

print("Your GPU Runtime is ready for action, Proceeding!")
print(100*'#')

model = "claude-3-5-sonnet-20240620"
model = "gpt-4o"

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": [model]
    }
)

gpt4_config = {
    "cache_seed": 42,
    "temperature": 0.4,
    "config_list": config_list_gpt4,
    "timeout": 600,
}

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt4_config,
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Give detailed pip dependencies first and then the python code, structure code blocks properly. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response, unless second one in dependency installation. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
Don't suggest running code that runs too long only suggest to work for development also don't run code that involves UI action like plt.show prefer saving the file. Code executor is enabled to perform GPU execution you can first fetch results of nvidia-smi and get the info of GPU attached.

Don't try to run code that is long/forever running or stalling for stdin like fastapi, plt.show(), if task is very big reduce the scope to make it easy. 

Always provide python pip dependencies first and then code.

Code provided is executed in a unique session with random filename and is automatically executed as python filename.py don't followup with any command on how to run the file.

Since code/files are not preserved between execution always generate end to end code to allow complete execution.
""",
)

scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=gpt4_config,
    system_message="Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.",
)

planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
Use RAG proxy agent to retrieve information stored and use it to better suggest data.
""",
    llm_config=gpt4_config,
)

executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "executor": monster_executor
    },
)

# RAG-enabled agent for content retrieval
rag_assistant = RetrieveUserProxyAgent(
    name="RAGAssistant",
    system_message="RAG-enabled assistant. Retrieve relevant information from the knowledge base to assist the team.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
            "https://raw.githubusercontent.com/unslothai/unsloth/main/README.md"
        ],
        "chunk_token_size": 1000,
        "model": config_list_gpt4[0]["model"],
        "vector_db": "chroma",
        "overwrite": True,
        "get_or_create": True,
    },
    code_execution_config=False
)

def retrieve_content(
    message: str,
    n_results: int = 3
) -> str:
    rag_assistant.n_results = n_results
    _context = {"problem": message, "n_results": n_results}
    ret_msg = rag_assistant.message_generator(rag_assistant, None, _context)
    return ret_msg or message

# Register the retrieve_content function for LLM agents
for agent in [engineer, scientist, planner]:
    agent.register_for_llm(
        description="Retrieve content for code generation and question answering.",
        api_style="function"
    )(retrieve_content)

# Register the retrieve_content function for execution in user_proxy and planner
for executor in [user_proxy, planner]:
    executor.register_for_execution()(retrieve_content)

groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, scientist, planner, executor, rag_assistant],
    messages=[],
    max_round=40
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

user_proxy.initiate_chat(manager, message=message)
