from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, GroupChat

from MonsterRuntimeAgent.MonsterRuntimeCodeExecutor  import MonsterRemoteCommandLineCodeExecutor

import os

monster_executor = MonsterRemoteCommandLineCodeExecutor(remote_url="http://localhost:8000")

code_execution_config={"executor": monster_executor}

llm_config={"config_list": [{'model': 'gpt-4', 'api_key': os.environ.get("OPENAI_API_KEY"), 'api_type': 'openai'}]}

#assistant = AssistantAgent("assistant", llm_config = llm_config)
coder = AssistantAgent(
    name="Coder",
    llm_config=llm_config,
)
user_proxy = UserProxyAgent("user_proxy", code_execution_config = code_execution_config) # IMPORTANT: set to True to run code in docker, recommended

""" # Example single agent generic
#assistant = AssistantAgent("assistant", llm_config = llm_config)
#user_proxy = UserProxyAgent("user_proxy", code_execution_config = code_execution_config) # IMPORTANT: set to True to run code in docker, recommended
#user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
"""

# Create PM agent to research papers and find results
pm = AssistantAgent(
    name="Team Lead",
    system_message="Reads through research papers and guider coder to develop POC and solve and optimize.",
    llm_config=llm_config,
)
groupchat = GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager, message="Find latest papers, github and blogs about LLM pruning and try on SLMs and suggest a code!"
)

