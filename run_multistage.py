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

cmodel = "claude-3-5-sonnet-20240620"
model = "gpt-4o" 

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
            Use 'UPDATE REQUIRED' to request changes or updates to the current plan or implementation.""",
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
        self.user_proxy.initiate_chat(self.manager, message=self.tree_of_throughts_plan)

    def get_planner_summary(self):
        history = self.summarizer.chat_messages[self.summarizer]
        for i in history:
            if i["name"].lower() == "summarizer":
                planning_summary = i["content"]
            else:
                pass
        
        upgraded_prompt = """
        Tree of thoughts plan:
        {self.tree_of_throughts_plan}

        Planning phase summaru:
        {planning_summary}
        """
        return upgraded_prompt


if __name__ == "__main__":
    print(100*'#')
    print(100*'#')
    print("Welcome to NeoV2 MonsterAPI Research Agent!\nI have a team of Engineer, GPU Code Executor, Research Scientist, Planner and a Critic! Go ahead and give me a AIML Development task!\n ")
    print(100*'#')

    path = "MonsterRuntimeAgent/competitions/dog-breed-prediction.md"

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
