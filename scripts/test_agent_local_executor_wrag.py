import sys
import time
import autogen
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from typing_extensions import Annotated

sys.path.append("/home/dev/MDockerRuntimeAPI/MonsterRuntimeAgent/")


from tools import MonsterNeoCodeRuntimeClient
from MonsterRuntimeCodeExecutor import MonsterRemoteCommandLineCodeExecutor

print(100*'#')
print(100*'#')
print("Welcome to NeoV2 MonsterAPI Research Agent!\n I have a team of Engineeer, GPU Code Executor, Research Scientist, Planner and a Critic! Go ahead and give me a AIML Development task!\n ") 
print(100*'#')

message = input("Enter Your Task here:")

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
    "timeout": 120,
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
Dont suggest  running code that runs too long only suggest to work for development also dont run code that involves UI action like plt.show prefer saving the file. Code executor is enabled to perform GPU execution you can first fetch results of nvidia-smi and get the info of GPU attached.

Dont try to run code that is long/forever running or stalling for stdin like fastapi, plt.show(), if task is very big reduce the scope to make it easy. 

Always provide python pip depencies first and then code.

Code provided is executed in a unqiue session with random filename and is automatically executed as python filename.py dont followup with any ccommand on how to run the file.

Since code/files are not preserved between execution always generate end to end code to allow complete execution.

only farm data from hugginface not from any source that requires a propreteiry API key. 

""",
)
scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=gpt4_config,
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.""",
)
planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
use RAG proxy agent to retreive information stored and use it to better suggest data.
""",
    llm_config=gpt4_config,
)
executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "use_docker": True
        #"executor":monster_executor
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

# RAG-enabled agent (knowledge_aid) for content retrieval
knowledge_aid = RetrieveUserProxyAgent(
    name="RAGAgent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
            "https://raw.githubusercontent.com/unslothai/unsloth/main/README.md"
        ],
        "chunk_token_size": 250,
        "model": config_list_gpt4[0]["model"],
        "vector_db": "chroma",
        "overwrite": True,
        "get_or_create": True,
    },
    code_execution_config=False
)

def _reset_agents():
    if user_proxy:
        user_proxy.reset()
    
    if engineer:
        engineer.reset()

    if scientist:
        scientist.reset()

    if planner:
        planner.reset()

def call_rag_chat():
    _reset_agents()

    # Content retrieval function for code generation or question answering
    def retrieve_content(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "Number of results to retrieve."] = 3
    ) -> str:
        knowledge_aid.n_results = n_results  # Set number of results
        _context = {"problem": message, "n_results": n_results}
        ret_msg = knowledge_aid.message_generator(knowledge_aid, None, _context)
        return ret_msg or message

    # Disable human input for knowledge_aid as it's for retrieving content only
    knowledge_aid.human_input_mode = "NEVER"

    # Register the retrieve_content function for LLM agents
    for caller in [planner, scientist]:
        d_retrieve_content = caller.register_for_llm(
            description="Retrieve content for code generation and question answering.",
            api_style="function"
        )(retrieve_content)

    # Register the retrieve_content function for execution in boss and pm
    for executor_ in [user_proxy, scientist]:
        executor_.register_for_execution()(d_retrieve_content)

    # Set up the group chat with agents
    groupchat = autogen.GroupChat(
        agents=[user_proxy, planner, scientist, engineer, executor],
        messages=[],  # Initialize with an empty message list
        max_round=40,
        speaker_selection_method="round_robin",  # Rotate speaker turn
        allow_repeat_speaker=False  # Prevent same agent from speaking consecutively
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

    # Optionally, you could return the groupchat object or initiate a chat directly
    return manager

# Example function call
if __name__ == "__main__":
    # Initialize the RAG-enabled conversation
    manager = call_rag_chat()

    # You can initiate the conversation with a specific message if needed:
    user_proxy.initiate_chat(manager, message=message)
