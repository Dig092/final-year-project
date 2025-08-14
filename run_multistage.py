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

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import chromadb

cmodel = "claude-3-5-sonnet-20240620"
model = "gpt-4o" 

config_list_gemini = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gemini-1.5-flash"]
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
MODE = "GPU"
if MODE == "CPU":
    sand_box = "Consider that CPU is only with 4GB RAM and reduce batch size and dataset size to fit and run faster on this CPU container."
else:
    sand_box = "Consider that GPU is only with 40GB GPU VRAM and reduce batch size and dataset size to fit and run faster on GPU."

def create_agent(name, system_message, llm_config):
    return autogen.AssistantAgent(name = name, system_message = system_message, llm_config = llm_config)

plannerphase_planner_system_message = """
You are planner that will verify vet and execute plan provided to you by user/admin.Work with critic on improving the solution.

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
"""
plannerphase_critic_system_message = """
Work as critic and criticise and improve planner and lead scientist final plan.
"""
pannerphase_lead_scientist_system_message = """
Work with critic and planner to make sure to provide feasible solution or execution plan.
perform internet search for required research as needed to solve any contention and make sure solution is top-notch.
"""
embeddings = OpenAIEmbeddings()
persistent_client = chromadb.PersistentClient("conversation_history")
collection  = persistent_client.get_or_create_collection("collection-1")
vectorstore = Chroma(client=persistent_client,collection_name="collection-1",embedding_function=embeddings)

def add_to_scratchpad(message:dict):
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    docs = [Document(page_content=message["content"],metadata={"speaker":message["name"].lower()})]
    retriever.add_documents(docs)

def retrieve_from_scratchpad(query:str)->str:
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    return (retriever | format_docs).invoke(query)

class InitialPlanner():
    def __init__(self, problem_statement):
        self.original_problem_statement = problem_statement
        self.tree_of_throughts_plan = self.create_tot_problem_statement()
        self.create_required_agents()
        self.register_function_calls()
        self.setup_groupchat()
        self.initiate_chat()

    def create_tot_problem_statement(self) -> str:
        planner =  ExperimentPlanner()
        tot_plan = planner.plan_from_prompt(prompt=self.original_problem_statement)
        return tot_plan

    def create_required_agents(self):
        """
        Planner, lead scientist and critic
        """
        self.admin =  autogen.UserProxyAgent(
            name="Admin",
            system_message="""A human admin. Interact with the planner, critic and scientist to discuss the plan. Plan execution needs to be approved by this admin.
            Use 'APPROVED' to indicate final approval of a plan or results.
            Use 'UPDATE REQUIRED' to request changes or updates to the current plan or implementation. End with summarizer summarizing the solution""",
            code_execution_config=False,
            human_input_mode="ALWAYS"
            )
        self.user_proxy =  autogen.UserProxyAgent(
            name="user_proxy",
            system_message="""user proxy  to perform require function calls to aid scientist.""",
            code_execution_config=False,
            human_input_mode="NEVER"
        )
        self.planner = create_agent("Planner", system_message=plannerphase_planner_system_message, llm_config=gpt4_config)
        self.critic = create_agent("Critic", system_message = plannerphase_critic_system_message, llm_config = claude_config)
        self.lead_scientist = create_agent("LeadScientist", system_message=pannerphase_lead_scientist_system_message, llm_config = claude_config)
        self.summarizer = create_agent("Summarizer", system_message="Summarize plan in detail along with refined problem statement solutions to follow and other critical conclusions from this planning group session dont include.", llm_config = claude_config)
    
    def register_function_calls(self):
        autogen.register_function(retreive_from_internet, caller=self.lead_scientist, executor=self.user_proxy, name="retreive_from_internet", description="Search internet and find context from internet.")

    def setup_groupchat(self):
        self.groupchat = autogen.GroupChat(
        agents=[self.user_proxy, self.planner, self.lead_scientist, self.critic, self.summarizer],
        messages=[],
        max_round=10,
        select_speaker_message_template = """You are in a role play game. The following roles are available:
                    {roles}.
                    Read the following conversation.
                    Then select the next role from {agentlist} to play. Only return the role.""",
        select_speaker_prompt_template = "Read the above conversation. Then select the next role from {agentlist} to play. Only return the role."
        )
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=gpt4_config)

    def initiate_chat(self):
        self.user_proxy.initiate_chat(self.manager, message=str(self.tree_of_throughts_plan))

    def get_planner_summary(self):
        history = self.manager._groupchat.messages
        planning_summary = ""
        for i in history[::-1]:
            add_to_scratchpad(i)
            if "name" in i and i["name"].lower() == "summarizer":
                planning_summary += i["content"]
            else:
                pass
        if planning_summary == "":
            print("Cannot parse plan summary!")
        
        upgraded_prompt = f"""
        Tree of thoughts plan:
        {self.tree_of_throughts_plan}

        Planning phase summary:
        {planning_summary}
        """
        return upgraded_prompt

data_management_guidelines = """
Dataset Management Guidelines:
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

    - Dependencies:
        - Always assume that Kaggle CLI and Huggingface CLI are already installed and are ready to be used. No need to install or set them up again.
"""

lead_data_engineer_system_message = f"""You are a Lead Data engineer. Dont use underscore when naming the agent. Look at the Plan provided by planning phase.
Plan how junior data engineer can load and transform the dataset as per the TOT plan.
Do not generate any code.
Plan the data engineering part of the pipeline.
Provide a clear unambigious step by step instruction to junior data engineer to write the code.
Also only make engineers use pytorch instead of tensorflow.

Always Remember:
{data_management_guidelines}
"""  

junior_data_engineer_system_message = f"""You are a junior data engineer and 
your task is to write code using the instructions provided by the lead data engineer.
You are also tasked with fixing bad code as per the suggestions given by the debugger.
Generate only the code for execution and nothing else.
Prefer OOPS and Modular programming for easier debugging and resuablity.
Also only make engineers use pytorch instead of tensorflow.

Always Remember:
{data_management_guidelines}
"""

executor_system_message = """You are the Executor responsible for running code and experiments. Your tasks include:
1. Executing code written by the Engineer in a controlled environment.
2. Reporting execution results accurately and completely.
3. Identifying and reporting any runtime errors or unexpected behaviors.
4. Monitoring resource usage (e.g., GPU memory, execution time) during code execution.
5. Providing performance metrics and system information when relevant.
6. Ensuring data integrity and proper handling of input/output operations.
7. Adhering to safety protocols when executing potentially risky code.
8. Maintaining a clean execution environment between runs to prevent interference."""

debugger_system_message = f""" You are a debugger whose task is to debug the code generated by junior data engineer and suggest fixes.
read the traceback carefully and try to understand what is causing the issue.
suggest exact fixes that needs to be done clearly.
Explain the junior data engineer why the code is failing and what needs to be fixed in order for the code to execute.
Also only make engineers use pytorch instead of tensorflow

Always Remember:
{data_management_guidelines}
"""



class DataEngineer():
    def __init__(self, problem_statement,executor):
        self.original_problem_statement = problem_statement
        self.executor = executor
        self.create_required_agents()
        self.register_function_calls()
        self.setup_groupchat()
        self.initiate_chat()

    def create_required_agents(self):
        """
        Planner, lead scientist and critic
        """
        self.admin =  autogen.UserProxyAgent(
            name="Admin",
            system_message="""A human admin. Interact with the Lead Data Engineer and Data engineer to discuss the plan to load and transform data. 
            Plan execution needs to be approved by this admin.
            Use 'APPROVED' to indicate final approval of a plan or results.
            Use 'UPDATE REQUIRED' to request changes or updates to the current plan or implementation.""",
            code_execution_config=False,
            human_input_mode="ALWAYS"
            )
        self.lead_data_engineer = create_agent("LeadDataEngineer", system_message=lead_data_engineer_system_message, llm_config=gpt4_config)
        self.junior_data_engineer = create_agent("JuniorDataEngineer", system_message = junior_data_engineer_system_message, llm_config = claude_config)
        self.executor = autogen.UserProxyAgent(name="Executor",system_message=executor_system_message,human_input_mode="NEVER",code_execution_config={"last_n_messages": 2,"executor": self.executor},)
        self.debugger = create_agent("Debugger",system_message=debugger_system_message,llm_config=claude_config)
        self.summarizer = create_agent("Summarizer", system_message="Summarize the execution details exclude the errors and exceptions occoured. Give details about how the data is loaded and where it is stored it's variable name etc,. Summary will be used by machine Leaning Engineer to use the loaded the data to train models", llm_config = gpt4_config)
    
    def register_function_calls(self):
        autogen.register_function(retreive_from_internet, caller=self.junior_data_engineer, executor=self.executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        autogen.register_function(retreive_from_internet, caller=self.debugger, executor=self.executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        autogen.register_function(retrieve_from_scratchpad, caller=self.lead_data_engineer, executor=self.executor, name="retreive_from_scratchpad", description="Search the scratchpad to find what happened before to further proceed.")
        autogen.register_function(retrieve_from_scratchpad, caller=self.junior_data_engineer, executor=self.executor, name="retreive_from_scratchpad", description="Search the scratchpad to find what happened before to further proceed.")
        autogen.register_function(retrieve_from_scratchpad, caller=self.debugger, executor=self.executor, name="retreive_from_scratchpad", description="Search the scratchpad to find what happened before to further proceed.")

    def setup_groupchat(self):
        self.groupchat = autogen.GroupChat(
        agents=[self.admin, self.lead_data_engineer, self.junior_data_engineer, self.executor,self.debugger,self.summarizer],
        messages=[],
        max_round=30,
        select_speaker_message_template = """You are in a role play game. The following roles are available:
                    {roles}.
                    Read the following conversation.
                    Then select the next role from {agentlist} to play. Only return the role.""",
        select_speaker_prompt_template = "Read the above conversation. Then select the next role from {agentlist} to play. Only return the role."
        )
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=gpt4_config)

    def initiate_chat(self):
        self.admin.initiate_chat(self.manager, message=self.original_problem_statement)

    def get_planner_summary(self):
        history = self.manager._groupchat.messages
        planning_summary = ""
        for i in history[::-1]:
            add_to_scratchpad(i)
            if i["name"].lower() == "summarizer":
                planning_summary += i["content"]
            else:
                pass
        if planning_summary == "":
            print("Cannot parse plan summary!")
        upgraded_prompt = f"""
            Data Engineering phase summary:
            {planning_summary}
            """
        return upgraded_prompt


training_code_guideline = """
Code Generation Guidelines for training or finetuning a model:
    - Always ensure that you write code for checkpointing the weights regularly (not too much) and saving the final weights after the process is completely executed.
    - The model checkpoints or weights must be stored in `/tmp/model` directory. If the directory doesn't exist then it must be created before storing the mdoels in it.
    - Ensure that the code has proper logging and formatting for each iteration/epoch.
"""

lead_machine_learning_engineer_system_message = f"""
You are a Lead ML Engineer and our task is to plan the training of models to beat4 the problem statement.
Data is already loaded and prepared for you from the data engineer. 
Based on the execution logs of the data engineer and plan given by the planner devise experiments to train models.
Do not generate any code.
Plan the Machine Learning part of the pipeline.
Provide a clear unambigious step by step instruction to junior machine learning engineer to write the code.

{training_code_guideline}
"""

junior_machine_learning_engineer_system_message = f"""
You are a junior Machine Learning Engineer. 
your task is to write code using the instructions provided by the lead Machine Learning engineer.
You are also tasked with fixing bad code as per the suggestions given by the debugger.
Once code works perfectly work alongside hyper parameter tuner to figure out better training parameters.
Generate only the code for execution and nothing else.
Prefer OOPS and Modular programming for easier debugging and resuablity.

Always Remember:
{training_code_guideline}
"""

ml_debugger_system_message = f""" You are a debugger whose task is to debug the code generated by junior machine learning engineer and suggest fixes.
read the traceback carefully and try to understand what is causing the issue.
suggest exact fixes that needs to be done clearly.
Eplain the junior data engineer why the code is failing and what nees to be fixed in order for the code to execute.

Always Remember:
{training_code_guideline}
"""

hyper_parameter_tuner_system_message = f"""You are an hyper parameter tuner whose job is to tune the 
hyper parameters to try and acheive a better convergence. 
Understand the results of the training and suggest better set of hyperparameters.
If the results are good enough just return NO HYPERPARAMETER TUNING REQUIRED.
Just return the required hyperpareters to be tuned in any case do not generate any code.
You should only suggest changes when the training happens successfully.

Always Remember:
{training_code_guideline}
"""

class MachineLearningEngineer():
    def __init__(self, problem_statement,tree_of_thougts_plan,executor):
        self.execution_journal = problem_statement
        self.tree_of_thoughts_plan = tree_of_thougts_plan
        self.executor = executor
        self.create_required_agents()
        self.register_function_calls()
        self.setup_groupchat()
        self.initiate_chat()

    def create_required_agents(self):
        
        self.admin =  autogen.UserProxyAgent(
            name="Admin",
            system_message="""A human admin. Interact with the ML engineer to trin your model. 
            Plan execution needs to be approved by this admin.
            Use 'APPROVED' to indicate final approval of a plan or results.
            Use 'UPDATE REQUIRED' to request changes or updates to the current plan or implementation.""",
            code_execution_config=False,
            human_input_mode="ALWAYS"
            )
        self.lead_machine_learning_engineer = create_agent("LeadMachineLearningEngineer", system_message=lead_machine_learning_engineer_system_message, llm_config=gpt4_config)
        self.junior_machine_learning_engineer = create_agent("JuniorMachineLearningEngineer", system_message =  junior_machine_learning_engineer_system_message, llm_config = claude_config)
        self.executor = autogen.UserProxyAgent(name="Executor",system_message=executor_system_message,human_input_mode="NEVER",code_execution_config={"last_n_messages": 2,"executor": self.executor},)
        self.debugger = create_agent("Debugger",system_message=ml_debugger_system_message,llm_config=claude_config)
        self.hyperparam_tuner = create_agent("HyperParameterTuner", system_message=hyper_parameter_tuner_system_message, llm_config = claude_config)
    
    def register_function_calls(self):
        autogen.register_function(retreive_from_internet, caller=self.junior_machine_learning_engineer, executor=self.executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        autogen.register_function(retreive_from_internet, caller=self.debugger, executor=self.executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        autogen.register_function(retreive_from_internet, caller=self.hyperparam_tuner, executor=self.executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        autogen.register_function(retrieve_from_scratchpad, caller=self.junior_machine_learning_engineer, executor=self.executor, name="retreive_from_scratchpad", description="Search the scratchpad to find what happened before to further proceed.")
        autogen.register_function(retrieve_from_scratchpad, caller=self.lead_machine_learning_engineer, executor=self.executor, name="retreive_from_scratchpad", description="Search the scratchpad to find what happened before to further proceed.")
        autogen.register_function(retrieve_from_scratchpad, caller=self.debugger, executor=self.executor, name="retreive_from_scratchpad", description="Search the scratchpad to find what happened before to further proceed.")

    def setup_groupchat(self):
        self.groupchat = autogen.GroupChat(
        agents=[self.admin, self.lead_machine_learning_engineer, self.junior_machine_learning_engineer, self.executor,self.debugger,self.hyperparam_tuner],
        messages=[],
        max_round=50,
        select_speaker_message_template = """You are in a role play game. The following roles are available:
                    {roles}.
                    Read the following conversation.
                    Then select the next role from {agentlist} to play. Only return the role.""",
        select_speaker_prompt_template = "Read the above conversation. Then select the next role from {agentlist} to play. Only return the role."
        )
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=gpt4_config)

    def initiate_chat(self):
        prompt = f"""Data Engineering execution journal:
        {self.execution_journal}
        
        Tree of Thoughts Plan:
        {self.tree_of_thoughts_plan}
        """
        self.admin.initiate_chat(self.manager, message=prompt)



if __name__ == "__main__":
    print(100*'#')
    print(100*'#')
    print("Welcome to NeoV2 MonsterAPI Research Agent!\nI have a team of Engineer, GPU Code Executor, Research Scientist, Planner and a Critic! Go ahead and give me a AIML Development task!\n ")
    print(100*'#')

    path = "MonsterRuntimeAgent/competitions/tweet-sentiment-extraction.md"

    message = open(path).read()
    # message = input("Enter Your Task here:")

    print(100*'#')
    print(100*'#')
    print("Let me give you a GPU Runtime!")
    print(".")
    time.sleep(1)
    print(".")

    client = MonsterNeoCodeRuntimeClient(container_type=MODE.lower(), cpu_count=16, memory = 32)
    monster_executor = MonsterRemoteCommandLineCodeExecutor(client=client)

    print("Your GPU Runtime is ready for action, Proceeding!")
    print(100*'#')

    planner = InitialPlanner(problem_statement=message)
    plan = planner.get_planner_summary()
    data_engineer = DataEngineer(problem_statement=plan,executor=monster_executor)
    data_engineering_execution_journal = data_engineer.get_planner_summary()
    ml_engineer = MachineLearningEngineer(problem_statement=data_engineering_execution_journal,tree_of_thougts_plan=planner.tree_of_throughts_plan,executor=monster_executor)
