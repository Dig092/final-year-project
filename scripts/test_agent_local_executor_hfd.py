import os
import sys
import time
import autogen

sys.path.append("/home/dev/MDockerRuntimeAPI/MonsterRuntimeAgent/")


from tools import MonsterNeoCodeRuntimeClient
from autogen import register_function
from MonsterRuntimeCodeExecutor import MonsterRemoteCommandLineCodeExecutor
from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
from autogen.agentchat.contrib.capabilities.transforms import  MessageHistoryLimiter, MessageTokenLimiter
from autogen.agentchat.contrib.capabilities.teachability import Teachability

from HFDatasetScraper import search_datasets_tool, get_summary_tool, DatasetExpertAgent


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
    "timeout": 600,
}
claude_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.4,
    "config_list": config_list_claude,
    "timeout": 3000,
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
COnsider that GPU is only with 40GB GPU VRAM and reduce batch size and dataset size to fit and run faster on GPU>

Always provide python pip depencies first and then code.

Code provided is executed in a unqiue session with random filename and is automatically executed as python filename.py dont followup with any ccommand on how to run the file.

Since code/files are not preserved between execution always generate end to end code to allow complete execution.

If required use hf dataset tool get_summary_tool to fetch datasets from hugginface along with summary only if needed and use the info as you see for engineering task.

""",
)

#register_function(search_datasets_tool, caller=engineer, executor=user_proxy, name="search_datasets", description="Search for datasets.")



teachability = Teachability(
            verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
                reset_db=True,
                    path_to_db_dir="./tmp/notebook/teachability_db",
                        recall_threshold=0.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
                        )

# Now add the Teachability capability to the agent.
teachability.add_to_agent(engineer)

#transform_messages.add_to_agent(engineer)

scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=claude_config,
    system_message="""Lead Scientist. You work as planner and drive the team based on admin needs, suggest and guide executor and engineer. You are able to categorize papers after seeing their abstracts printed. You suggest code improvements to engineer and coder, make sure to reduce scope of experimentation to suit the POC scope and perform experiments. When using large model datasets consider GPU vram with only 40GB and reduce dataset size, batch size and reduce scope to fit and run faster as POC.""",
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
        "last_n_messages": 1,
        #"use_docker": True
        "executor":monster_executor
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

register_function(get_summary_tool, caller=engineer, executor=executor, name="get_summary", description="Get a search summary of datasets.")

groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, scientist, executor], messages=[], max_round=40
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

user_proxy.initiate_chat(
        manager, message=message
)
