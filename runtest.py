import os
import sys
import time
import json
import autogen
from typing import Optional, Dict, Any, List, Union

from autogen import register_function
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter, MessageTokenLimiter
from autogen.agentchat.contrib.capabilities import teachability

from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor

from autogen import token_count_utils
from autogen.cache.in_memory_cache import InMemoryCache
from autogen.oai.openai_utils import filter_config

from MonsterRuntimeAgent.MonsterRuntimeCodeExecutor import MonsterRemoteCommandLineCodeExecutor
from MonsterRuntimeAgent.Tools.RuntimeTools import MonsterNeoCodeRuntimeClient
from MonsterRuntimeAgent.Tools.ExperimentationModel import ExperimentPlanner
from MonsterRuntimeAgent.Tools.HFDatasetScraper import get_summary_tool
from MonsterRuntimeAgent.Tools.NetScraper import retreive_from_internet
from MonsterRuntimeAgent.Tools.StateTrackingClass import ATSTool
from MonsterRuntimeAgent.Tools.StateManager import init_ats, update_stage_status, add_subtask, update_subtask, get_stage_status

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import chromadb
import logging
from datetime import datetime
import tiktoken

# Define a function to generate a unique log filename each run
def generate_log_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"logs/runtime_{timestamp}.log"

# Ensure the logs directory exists
os.makedirs("autogen_logs", exist_ok=True)
os.makedirs("autogen_logs/logs", exist_ok=True)

log_filename = generate_log_filename()
# logging_session_id = autogen.runtime_logging.start(logger_type="file", config={"filename": log_filename})

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

config_list_claude_opus = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["claude-3-opus-20240229"]
    }
)

config_list_o1 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["o1-preview"]
    }
)

config_list_gemini_flash = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gemini-2.0-flash-exp"]
    }
)

gpt4_config = {
    "cache_seed": 42,
    "temperature": 0.4,
    "max_retries": 4,
    "config_list": config_list_gpt4,
    "timeout": 1000,
    "max_tokens": 12000
}
claude_config = {
    "cache_seed": 42,
    "temperature": 0.4,
    "max_retries": 15,
    "config_list": config_list_claude,
    "timeout": 30000,
}
o1_config = {
    "cache_seed": 42,
    "config_list": config_list_o1,
    "timeout": 30000,
}
claude_opus_config = {
    "cache_seed": 42,
    "temperature": 0.4,
    "max_retries": 5,
    "config_list": config_list_claude_opus,
    "timeout": 30000,
}

gemini_config = {
    "cache_seed": 42,
    "max_retries": 5,
    "config_list": config_list_gemini,
    "timeout": 30000,
}

gemini_flash_config = {
    "cache_seed": 42,
    "max_retries": 5,
    "config_list": config_list_gemini_flash,
    "timeout": 30000,
}

MODE = "GPU"
if MODE == "CPU":
    sand_box = "Consider that CPU is only with 4GB RAM and reduce batch size and dataset size to fit and run faster on this CPU container."
else:
    sand_box = "Consider that GPU is only with 40GB GPU VRAM and reduce batch size and dataset size to fit and run faster on GPU."

def create_agent(name, system_message, llm_config):
    return autogen.AssistantAgent(name = name, system_message = system_message, llm_config = llm_config)

# Initialize LLMLingua compressor
llm_lingua = LLMLingua()

# Create text compressor with specific settings
text_compressor = TextMessageCompressor(
    text_compressor=llm_lingua,
    compression_params={"target_token": 48000},  # Aggressive compression target
    cache=None
)

def create_gpt_agent(name, system_message, llm_config):
    agent = autogen.AssistantAgent(name = name, system_message = system_message, llm_config = llm_config)
    # Create and apply message transformer
    message_transformer = transform_messages.TransformMessages(
        transforms=[text_compressor]
    )
    message_transformer.add_to_agent(agent)
    return agent

embeddings = OpenAIEmbeddings()
persistent_client = chromadb.PersistentClient("conversation_history")
collection  = persistent_client.get_or_create_collection("collection-1")
vectorstore = Chroma(client=persistent_client,collection_name="collection-1",embedding_function=embeddings)

def addToScratchpad(message:dict):
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    docs = [Document(page_content=message["content"],metadata={"speaker":message["name"].lower()})]
    retriever.add_documents(docs)

def retrieveFromScratchpad(query: str, max_tokens: int = 8000) -> str:
    """
    Retrieves context from vectorstore and summarizes it to stay within token limit (Max tokens limit: 8000)
    """
    # First retrieve the context
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    context = (retriever | format_docs).invoke(query)

    compression_config = TextMessageCompressor(
        text_compressor=llm_lingua,
        compression_params={"target_token": max_tokens},  # Aggressive compression target
        cache=None
    )

    # Count tokens in initial context
    initial_tokens = len(tokenizer.encode(context))
    # If already within limit, return as is
    if initial_tokens <= max_tokens:
        return context
    
    # Compress the context using LLM Lingua
    compressed_text = compression_config.apply_transform([{"content": context}])    
    return compressed_text[0]["content"]

class TOTFunc:
    def __init__(self):
        self.tot_final_plan = None
    
    def generate_tot(self, problem_input):
        # Example processing - you can modify this based on your needs
        self.planner  = ExperimentPlanner()
        self.tot_final_plan = self.planner.plan_from_prompt(prompt=problem_input)
        return self.tot_final_plan
    
    def access_tot(self):
        return self.tot_final_plan

tot_functions = TOTFunc()


def create_tot_problem_statement(problem_statement:str):
    """Create implementation plan using tree of thoughts thinking process based on provided context.
    Args:
        problem_statement (str): A string containing the provided message or context.
    """
    tot_plan = tot_functions.generate_tot(problem_statement)
    
    print("#"*100)
    print(f"TOT Plan: {tot_plan}")
    print("#"*100)
    return tot_plan

def get_tot_problem_statement():
    """Access the refined tree of thoughts implementation plan."""
    tot_plan = tot_functions.access_tot()
    return tot_plan

example_code_ats = """
```python
# Update stage status
update_stage_status(stage_number=1, stage_name="Dataset Engineering & Preparation")

# Add subtask
add_subtask(stage_number=1, subtask_name="Download dataset")

# Get status
status = get_stage_status()

update_stage_status(stage_number=1, stage_name="Dataset Engineering & Preparation", stage_accomplished="No", stage_summary="Started Dataset Engineering & Preparation")

# Add a subtask
add_subtask(stage_number=1, subtask_name="Download dataset", summary="Implemented code for downloading the provided dataset. Testing now.")

# Update a subtask
update_subtask(stage_number=1, subtask_index=0, accomplished="Yes", summary="Completed requirements gathering")
update_subtask(stage_number=1, subtask_index=0, accomplished="No")

# Check full status
status = get_stage_status()
```
"""

teamplanner_system_message = f"""
You are team planner for ML Engineering team.

You need to understand the problem statement / task at a deeper level from ML engineering standpoint.

You can use "generate_tree_of_thought_plan" tool to generate a structured executable plan using the context provided by user/admin. 

You are also equipped with an Automated Tracking System (ATS) tool to maintain conversation stages and subtasks. 
You can update stages, manage subtasks, and check status using the provided functions.
    
Based on the generated tree of though plan, create stages and detailed subtasks in each stage for the team using the ATS tool.

Avoid repetitive subtasks and combine them together if possible. 
Keep 5 max subtasks for each stage. 
Keep each stage segmented properly so the subtasks are not common.

An example of stages and their typical subtasks for an MLE problem could be like:
    1. Data Engineering & Preparation:
        - Data collection and storage (path details etc)
        - Exploratory data analysis, Data cleaning and preprocessing
        - Feature engineering and selection
        - Data validation and quality checks
        - Creating data pipelines
        - Managing data versioning

    2. Model Development & Training:
        - Model architecture design
        - Training workflows
        - Cross-validation strategies
        - Experiment tracking
        - Initial model evaluation
        - Development environment setup

    3. Model Optimization & Enhancement:
        1. Performance Analysis against the target metric/goal:
            - Calculate baseline metrics
            - Identify top 3 bottlenecks
            - Set optimization targets
        2. Initial Optimization Experiments:
            - Learning rate and batch size tuning
            - Model pruning and quantization
            - Data augmentation enhancement
            - Evaluate results against target
        3. Advanced Optimization (if needed):
            - Knowledge distillation
            - Architecture modifications
            - Ensemble methods
        4. Final Validation:
            - Compare with target metrics
            - Document improvements

This list is just an example. The actual stages and subtasks would vary depending on the problem statement / task provided by admin. Prepare accordingly. 

The stages or subtasks must include context relevant for completion of the problem such as target eval metrics, target outcomes etc. (if provided).

If a task needs to build a submission based on evaluation for the trained model then put training and evaluation tasks together in the same sub task. 
Ensure that the target is achieved before accomplishing all the stages. Push the ML team to improve further if the target is not met.

Example Usage of ATS tool to add stages or sub-tasks and update their status:
{example_code_ats}

Keep tracking all the updates from team members and add specific details to help the other team members as and when needed.

Upon accomplishing all the stages, use "retrieve_tree_of_thought_plan" tool to evaluate if the target or the required goals have been achieved? 
If not achieved then iterate on the subtasks in specific stages where further improvements can be made.

Depending on the given task or additional information provided by the user/admin, you'd need to re-align with the problem statement and improve the context for the specific stages and readjust their sub-tasks and statuses if needed.
If admin doesn't provide any constructive feedback then think yourself and iterate for the next steps accordingly.
"""

"""
validate the status with 
If the target is still not achieved after accomplishing all the stages, then we should keep on readjusting the subtasks in stages to further improve and achieve the final target. 
"""

executor_system_message = """You are the Executor responsible for running code and experiments. Your tasks include:
1. Executing code written by the Engineer in a controlled environment.
2. Reporting execution results accurately and completely.
3. Identifying and reporting any runtime errors or unexpected behaviors.
4. Monitoring resource usage (e.g., GPU memory, execution time) during code execution.
5. Providing performance metrics and system information when relevant.
6. Ensuring data integrity and proper handling of input/output operations.
7. Adhering to safety protocols when executing potentially risky code.
8. Maintaining a clean execution environment between runs to prevent interference.
9. If no code is provided then reply "No code provided to execute".
10. If tool to be executed doesn't exist then reply "Tool call invalid. Tool does not exist".
"""

gpu_optimizations = """
When writing GPU-accelerated code:

1. Accelerated Libraries:
   - Use NVIDIA RAPIDS (cuDF, cuML) for data processing
   - Consider NVIDIA DALI for GPU data loading
   - Prefer torch.cuda operations over NumPy
   - Use Numba for custom CUDA kernels
   - Use cupy for GPU-accelerated numpy operations

2. Data Pipeline:
   - Maximize batch size for GPU memory
   - Use multiple workers in DataLoader
   - Enable pin_memory and prefetch_factor
   - Cache dataset indices/paths
   - Use non_blocking transfers

3. GPU Optimization:
   - Enable cudnn.benchmark
   - Use Automatic Mixed Precision (amp)
   - Clear GPU cache regularly
   - Monitor GPU utilization
   - Use gradient scaling
   - Zero_grad with set_to_none=True

4. Memory Management:
   - Move data transforms to GPU where possible
   - Minimize CPU-GPU transfers
   - Use efficient data formats
   - Profile memory usage
"""

lead_data_engineer_system_message = f"""
You are a lead Data engineer in ML team. Your aim is to ensure that the data engineer is working properly and completing all the necessary tasks under your supervision.

Your focus has to be on Data Engineering & Preparation stage related tasks only.
Coordinate with Data engineer and provide the necessary next steps and tasks to be completed. 
Push Data engineer to always execute any code generated by it to ensure that it has executed the necessary steps. 
Upon successful completion of each task by Data engineer, report a status update to the Planner.
Upon failure of a task, work with Data engineer to help resolve the issue.

Use `get_stage_status` tool to fetch all the stages and subtasks prepared by the Planner. 
Use `update_stage_status` tool to update the stage status only when all the subtasks of that stage have been accomplished.
Use `add_subtask` tool to add new sub-tasks needed to accomplish the Data engineering stage for achieving the target.
Use `update_subtask` tool to update the status of a subtask only when the task has been successfully completed or failed.
Example usage of tools - get_stage_status, update_stage_status, add_subtask and update_subtask: 
{example_code_ats}

Do not proceed with the next sub-tasks until previous sub tasks in the data engineering stages are complete and marked their "accomplished" status as "yes".

Reporting Guidelines:
    - Update subtasks sequentially as the conversation progresses.
    - Provide summaries for the stage and sub-tasks to Planner that are concise but informative.
    - Mark stages as accomplished only when the sub-tasks are fully met.
    - Always consider the conversation context when updating stages.
    - Ensure that the data engineer has completed the requested sub-tasks before updating the status.

Report to Planner once the data stage has been completed successfully.

NOT TO DO:
- DO NOT GENERATE ANY CODE.
- DO NOT PROVIDE ANY CODE OUTLINES. ONLY DATA ENGINEER MUST GENERATE THE CODE AND EXECUTE IT.
- DO NOT PROVIDE TRAINING OR HYPERPARAMETER TUNING TASKS TO DATA ENGINEER.
"""

data_management_guidelines = """
Dataset Management Guidelines:
    - Download Location:
        - Always download and store data in the `/tmp/data/` directory.

    - Always generate code in python language only. Use code blocks with language specification, e.g., ```python``` seperator file will be run automatically next no need to specify run command. 

    - Handling Nested Compressed Files:
        - After downloading a dataset (e.g., `abc.zip`) or accessing a local dataset directory, automatically detect and recursively unzip any compressed files until all data files are extracted.
        - For 7z compressed files, use p7zip package. Example code for extracting a .7z file: ```bash 7za x myarchive.7z```.
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

Give me the full code, so I can copy and paste it on one go. Do not summarise things like //rest of function here. The intent is so I Can copy and paste things seamslessly, since I am very lazy.
"""

data_engineer_system_message = f"""
You are the core data engineer in ML team.

Only focus on implementing the code and get it executed for the tasks provided by Lead Data Engineer.

Guidelines for some type of tasks:
1. Perform Dataset Analysis and Acquisition:
    - If a dataset is provided:
        - Write efficient Python or Bash scripts to download, parse, and manage the dataset.
        - Perform exploratory data analysis (EDA) to understand the dataset's structure and representation.

    - If a dataset is not provided in the task, only then:
        - Identify relevant online datasets for the problem statement.
        - Use the `retreive_from_internet` tool to search for datasets online.
        - Utilize the `get_summary_tool` to obtain summaries of datasets from platforms like Hugging Face.
        
    - If a Kaggle dataset download command is provided, assume the Kaggle CLI is installed and proceed to download using the command.

2. Provide a data processing pipeline:
    - Prepare and manage datasets effectively for machine learning tasks, ensuring that all data is properly acquired, processed, and organized for subsequent steps in the AI development pipeline. Leverage your expertise to streamline data handling and provide clear, executable code solutions.
    - Based on this you can produce a final report that provides details on the data processing pipeline that includes details such as what is the dataset structure, where is it stored, code for using it and any other relevant information.

3. Once the data has been downloaded, extracted and stored in `/tmp/data`, generate the final code if needed for data processing pipeline that the ML engineering team can use. Store that file in /tmp/data_processing.py and provide that information further once saved.

Validate the output of the execution logs provided by executor and based on that for each completed or failed task, provide a summary and status update to Lead Data engineer. 
Only talk to Lead Data engineer after you have an execution update on a task, otherwise focus on improving and executing the task with executor.

If executor returns empty log then it means the code was not provided to executor. Provide the code properly so it can run it and provide your the execution logs.

Always Remember:
- Do not assume the file names or structure to explore datasets. Implement code to list all dataset files first if needed to understand the dataset file structure and then proceed ahead.
- Always provide complete code (combine all steps and avoid incomplete code).
- Optimize the code for multiprocessing and GPU based data processing.
- Prefer OOPS and Modular programming for easier debugging and resuablity.
- Always use pytorch instead of tensorflow.
- {data_management_guidelines}

Make use of GPU accelerations if possible in code:
{gpu_optimizations}

Not to do:
- Do not write code for training or finetuning tasks. You are not an ML engineer. 
- Do not provide incomplete code. 
"""

lead_machine_learning_engineer_system_message = f"""
You are a Lead Machine Learning Engineer in ML team. Your aim is to ensure that the ML Engineer and Hyper parameter tuners complete all the necessary tasks under your supervision.

Your focus has to be on Model Development & Training and Optimization stages related tasks only.
Coorindate with ML Engineer and Hyperparameter tuner to provide the necessary next steps and tasks to be completed.
Push ML engineer and Hyperparameter tuner to always execute any code generated by them to ensure that it has executed the necessary steps. 
Upon successful completion of each task by ML engineer or Hyperparameter tuner, report a status update to the Planner.
Upon failure of a task, work with ML engineer or Hyperparameter tuner to help resolve the issue.
In task summary try to mention useful details such as model save location, eval metrics outcome etc.

Use `get_stage_status` tool to fetch all the stages and subtasks prepared by the Planner. 
Use `update_stage_status` tool to update the stage status only when all the subtasks of that stage have been accomplished.
Use `add_subtask` tool to add new sub-tasks needed to accomplish the ML engineering stages for achieving the target.
Use `update_subtask` tool to update the status of a subtask only when the task has been successfully completed or failed.
Example usage of tools - get_stage_status, update_stage_status, add_subtask and update_subtask: 
{example_code_ats}

Do not proceed with the next sub-tasks until previous sub tasks in the ML engineering stages are complete and marked their "accomplished" status as "yes".

Data is already preprocessed for ML workloads by the data engineering team. Explore the needed data related files in /tmp/data.

If the goal requires to generate a submission based on evaluation for the trained model then put the training and evaluation tasks together in the same sub task.

Reporting Guidelines:
    - Update subtasks sequentially as the conversation progresses.
    - Provide summaries for the stage and sub-tasks to Planner that are concise but informative.
    - Mark stages as accomplished only when the sub-tasks are fully met.
    - Always consider the conversation context when updating stages.
    - Ensure that the ML engineer and hyperparameter tuner have completed the requested sub-tasks before updating the status.

NOT TO DO:
- DO NOT GENERATE ANY CODE.
- DO NOT PROVIDE ANY CODE OUTLINES. ONLY ML ENGINEER MUST GENERATE THE CODE AND EXECUTE IT.
- DO NOT PROVIDE TRAINING OR HYPERPARAMETER TUNING TASKS TO DATA ENGINEER.

Experiment guidelines:
- In the beginning always suggest to run very small experiments with early stopping as a starting point for the team.
- Based on the results of the smaller experiments, suggest the next steps to decide on models or hyperparameters.
- Make sure to optimize the batchsize for best/faster training and then proceed to complete model finetuning.
- Always suggest the team to use Pytorch, GPU based Multiprocessing data loading and training for faster execution.
- Iteratively keep exploring and improving the approach unless Target is achieved or unless Admin suggests to stop the process or do something else.

Use "retrieve_tree_of_thought_plan" tool to evaluate if the target or the required goals have been achieved or not. Regroup with planner and plan the next steps accordingly.

If the target has not been achieved and Planner suggests for improvements to achieve the target then incorporate the feedback and plan the next steps and coordinate with ML Engineer and HyperparameterTuner accordingly for code implementation and execution.

Model training suggestion guidelines:
- The model checkpoints or weights must be stored in `/tmp/model` directory.
- Suggest to use latest optimized architecture models to solve complex problem statements such as transformer models, ensemble models, latest Large Language models and NLP models if needed to maximize improvements.
- Suggest to use GPU for both data loading and training workloads for faster processing.
- Suggest to use Pytorch instead of Tensorflow.
- Suggest to avoid memory bottlenecks by including memory management (garbage collection, memory cleanup), gradient accumulation, batch size optimization, and early stopping mechanisms, and precompute/cache data where possible.
- Ask ML Engineer to always generate submission.csv along with each model training experiment in its code.
"""

model_preference_list = """
Here's a focused list of recommended architectures by task type:

Computer Vision:
1. Image Classification: EfficientNetV2 → ConvNeXt → Vision Transformer → Ensemble of these
2. Object Detection: YOLO (v8) → DETR → Cascade R-CNN → Ensemble with NMS
3. Segmentation: Mask2Former → SegFormer → DeepLabV3+ → Weighted ensemble

Natural Language Processing:
1. Text Classification: Deberta-v3 → RoBERTa → Ensemble of transformers
2. Question Answering: GPT-3.5/4 → PaLM → T5/FLAN-T5 → Ensemble with voting
3. Named Entity Recognition: Deberta → LUKE → Ensemble with majority voting
4. Translation: NLLB → mT5 → Ensemble with confidence weighting

Tabular Data:
1. Structured Prediction: LightGBM → CatBoost → XGBoost → Stacked ensemble
2. Time Series: LightGBM with time features → Transformer → Neural ODE → Weighted ensemble
3. Recommendation: DCN V2 → DeepFM → AutoInt → Ensemble of different architectures

Audio:
1. Speech Recognition: Whisper → Conformer → Wav2Vec 2.0 → Ensemble with confidence
2. Audio Classification: AST → HTS-AT → ResNet + spectrogram → Weighted ensemble

Multimodal:
1. Vision-Language: CLIP → BEiT-3 → CoCa → Ensemble with task-specific weighting
2. Audio-Visual: AV-HuBERT → ImageBind → Weighted ensemble

For ensembles, prefer:
- Stacking over averaging for complex tasks
- Diversity in base models (different architectures)
- Model-specific confidence scores for weighted voting
- Cross-validation for reliable stacking weights
"""

training_code_guideline = f"""
Important Code writing guidelines for Training or Fine-Tuning a Model:
* Write complete, executable code without omissions.
* Initialize model/optimizer once before training loop.
* Save model checkpoints and final weights in `/tmp/model` directory. Create directory if it doesn't exist.
* Avoid saving checkpoints during k-fold validation or hyperparameter optimization experiments unless the evaluation is suggesting positive outcome.
* Implement detailed logging of training or data loading progress and metrics.
* Implement early stopping and gradient accumulation.
* Include the necessary evaluation logic in the code.
* Optimize batch size and memory management.
* Use multiprocessing for data loading/processing.
* Include garbage collection and memory cleanup.
* Start with small validation experiments and increase experiment size as path gets cleared.
* Maintain model/optimizer state throughout training.
* Only load best checkpoint for final evaluation.
* Precompute and cache data where possible.
* Iteratively improve until target metrics achieved.
* Generate submission.csv along with each model training experiment.

DataLoader Configuration:
* Calculate optimal num_workers based on CPU cores (cpu_count()*1)
* Always enable pin_memory=True for GPU training
* Set persistent_workers=True for multi-epoch efficiency
* Include prefetch_factor for advance batch loading
* Monitor and log data loading times
* Never use num_workers=0 unless explicitly required

Never:
* Omit code sections or use placeholders
* Reload base model during training
* Leave out error handling or logging
* Skip memory optimization steps
* Write code in steps. Always write full complete code in one go.

If executor returns empty log then it means the code was not provided to executor. Provide the code properly so it can run it and provide your the execution logs.
"""

machine_learning_engineer_system_message = f"""
You are a Kaggle Grandmaster level expert ML engineer specializing in PyTorch and transformers.
Your task is to implement code and get it executed by the executor for the tasks provided by the Lead ML engineer.
You are also tasked with fixing bad code based on the error logs of executor.
Once code works perfectly work alongside hyper parameter tuner to figure out better training parameters.
Generate only the code for execution and nothing else.
Prefer OOPS and Modular programming for easier debugging and resuablity.

{training_code_guideline}

Optimize the code for faster performance:
{gpu_optimizations}

Model architecture preference list for specific tasks:
{model_preference_list}

Evaluate the execution logs from the executor and update the Lead ML Engineer only after successfully running the code and assessing the results.

Make sure you provide proper and complete code.

If executor returns empty log then it means the code was not provided to executor. Mention the code properly in your response so the executor can run it and provide your the execution logs.

Focus on executing tasks with the executor and improving until you have a clear status update. For hyperparameter tuning, consult the Hyperparameter Tuner for strategies and target parameters to optimize model performance.
"""

hyper_parameter_tuner_system_message = f"""
You are an expert hyper parameter tuner whose job is to tune the hyper parameters to try and acheive a better convergence. 
Just return the required optimal hyperparameters to be tuned, in any case do not generate any code.

Understand the results of the training and suggest better set of hyperparameters.
If the results are accomplishing the target (if mentioned) on smaller sample set then run it on complete data set.
If the results are good enough just return NO HYPERPARAMETER TUNING REQUIRED.
You should only suggest changes when the training happens successfully.

NOT TO DO:
- DO NOT GENERATE ANY CODE.

In case of any missing information or gaps such as what's the target to be achieved or current status, coordinate with ML engineer or Lead ML engineer.
"""

class InitialPlanner():
    def __init__(self, problem_statement,executor):
        self.original_problem_statement = problem_statement
        # self.tree_of_throughts_plan = self.create_tot_problem_statement()
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
            system_message="""A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.
            Use 'APPROVED' to indicate final approval of a plan or results.
            Use 'UPDATE REQUIRED' to request changes or updates to the current plan or implementation. End with summarizer summarizing the solution""",
            code_execution_config=False,
            human_input_mode="ALWAYS"
            )
        self.user_proxy =  autogen.UserProxyAgent(
            name="user_proxy",
            system_message="""user proxy to perform require function calls.""",
            code_execution_config=False,
            human_input_mode="NEVER"
        )
        self.planner = create_gpt_agent("Planner", system_message=teamplanner_system_message, llm_config=gpt4_config)
        self.executor = autogen.UserProxyAgent(name="Executor",system_message=executor_system_message,human_input_mode="NEVER",code_execution_config={"last_n_messages": 2,"executor": self.executor},)
        self.lead_data_engineer = create_agent("LeadDataEngineer", system_message = lead_data_engineer_system_message, llm_config = gpt4_config)
        self.data_engineer = create_agent("DataEngineer", system_message = data_engineer_system_message, llm_config = gpt4_config)
        self.lead_machine_learning_engineer = create_gpt_agent("LeadMLEngineer", system_message = lead_machine_learning_engineer_system_message, llm_config = gpt4_config)
        self.machine_learning_engineer = create_agent("MLEngineer", system_message = machine_learning_engineer_system_message, llm_config = claude_config)
        self.hyperparam_tuner = create_agent("HyperparameterTuner", system_message = hyper_parameter_tuner_system_message, llm_config = gpt4_config)
    
    def register_function_calls(self):
        autogen.register_function(retreive_from_internet, caller=self.planner, executor=self.executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        autogen.register_function(retreive_from_internet, caller=self.data_engineer, executor=self.executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        autogen.register_function(retreive_from_internet, caller=self.machine_learning_engineer, executor=self.executor, name="retreive_from_internet", description="Search internet and find context from internet.")
        
        autogen.register_function(get_summary_tool, caller=self.data_engineer, executor=self.executor, name="get_summary_tool", description="Obtain summaries of datasets from platforms like Hugging Face.")
        
        autogen.register_function(create_tot_problem_statement, caller=self.planner, executor=self.executor, name="generate_tree_of_thought_plan", description="Generate refined implementation plan using tree of thoughts thinking process based on provided context.")
        autogen.register_function(get_tot_problem_statement, caller=self.planner, executor=self.executor, name="retrieve_tree_of_thought_plan", description="Retrieve the tree of thoughts implementation plan.")
        autogen.register_function(get_tot_problem_statement, caller=self.lead_machine_learning_engineer, executor=self.executor, name="retrieve_tree_of_thought_plan", description="Retrieve the tree of thoughts implementation plan.")

        autogen.register_function(update_stage_status, caller=self.planner, executor=self.executor, name="update_stage_status", description="Add or Update a conversation stage in the ATS system")
        autogen.register_function(update_stage_status, caller=self.lead_data_engineer, executor=self.executor, name="update_stage_status", description="Add or Update a conversation stage in the ATS system")
        autogen.register_function(update_stage_status, caller=self.lead_machine_learning_engineer, executor=self.executor, name="update_stage_status", description="Add or Update a conversation stage in the ATS system")
        
        autogen.register_function(get_stage_status, caller=self.planner, executor=self.executor, name="get_stage_status", description="Get current status of all stages and their subtasks")
        autogen.register_function(get_stage_status, caller=self.lead_data_engineer, executor=self.executor, name="get_stage_status", description="Get current status of all stages and their subtasks")
        autogen.register_function(get_stage_status, caller=self.lead_machine_learning_engineer, executor=self.executor, name="get_stage_status", description="Get current status of all stages and their subtasks")
        
        autogen.register_function(add_subtask, caller=self.planner, executor=self.executor, name="add_subtask", description="Add a subtask to a specific stage")
        autogen.register_function(update_subtask, caller=self.planner, executor=self.executor, name="update_subtask", description="Update a specific subtask's status")
        autogen.register_function(add_subtask, caller=self.lead_data_engineer, executor=self.executor, name="add_subtask", description="Add a subtask to a specific stage")
        autogen.register_function(update_subtask, caller=self.lead_data_engineer, executor=self.executor, name="update_subtask", description="Update a specific subtask's status")
        autogen.register_function(add_subtask, caller=self.lead_machine_learning_engineer, executor=self.executor, name="add_subtask", description="Add a subtask to a specific stage")
        autogen.register_function(update_subtask, caller=self.lead_machine_learning_engineer, executor=self.executor, name="update_subtask", description="Update a specific subtask's status")

    def setup_groupchat(self):  
        self.groupchat = autogen.GroupChat(
        agents=[self.user_proxy, self.planner, self.executor, self.data_engineer, self.lead_data_engineer, self.lead_machine_learning_engineer, self.machine_learning_engineer, self.hyperparam_tuner],
        messages=[],
        max_round=250,
        allow_repeat_speaker=False,
        select_speaker_message_template = """You are in a role play game. The following roles are available:
                    {roles}.
                    Read the following conversation.
                    Then select the next role from {agentlist} to play. Only return the role. Always trigger summarizer only the end of task.
                    """,
        select_speaker_prompt_template = "Read the above conversation. Then select the next role from {agentlist} to play. Only return the role.",
        )
        self.groupchat.allowed_speaker_transitions_dict = {agent: [] for agent in self.groupchat.agents}
    
        transitions = {
                self.admin: [self.planner],
                self.planner: [self.admin, self.executor, self.lead_data_engineer, self.lead_machine_learning_engineer],
                self.lead_data_engineer: [self.data_engineer, self.planner],
                self.data_engineer: [self.executor, self.lead_data_engineer],
                # self.executor: [self.planner, self.data_engineer, self.machine_learning_engineer,q self.hyperparam_tuner],
                self.lead_machine_learning_engineer: [self.machine_learning_engineer, self.hyperparam_tuner, self.planner],
                self.machine_learning_engineer: [self.executor, self.hyperparam_tuner, self.lead_machine_learning_engineer],
                self.hyperparam_tuner: [self.executor, self.machine_learning_engineer, self.lead_machine_learning_engineer],
        }
            
        for agent, allowed_speakers in transitions.items():
                self.groupchat.allowed_speaker_transitions_dict[agent] = allowed_speakers
                
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=gemini_flash_config)

    def initiate_chat(self):
        self.user_proxy.initiate_chat(self.manager, message=self.original_problem_statement)

if __name__ == "__main__":
    print(100*'#')
    print(100*'#')
    print("Welcome to NeoV2 MonsterAPI Research Agent!\nI have a team of Engineer, GPU Code Executor, Research Scientist, Planner and a Critic! Go ahead and give me a AIML Development task!\n ")
    print(100*'#')
    import sys
    
    try:
        file_n = sys.argv[1]
    except IndexError:
        file_n = "tweet-sentiment-extraction.md"

    path = f"MonsterRuntimeAgent/competitions/{file_n}"

    message = open(path).read()
    # message = input("Enter Your Task here:")

    print(100*'#')
    print(100*'#')
    print("Let me give you a GPU Runtime!")
    print(".")
    time.sleep(1)
    print(".")

    # Instantiate the Teachability capability. Its parameters are all optional.
    teachability = teachability.Teachability(
        verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
        reset_db=True,
        path_to_db_dir="./tmp/notebook/teachability_db",
        recall_threshold=1.0,  # Higher numbers allow more (but less relevant) memos to be recalled.
    )

    client = MonsterNeoCodeRuntimeClient(container_type=MODE.lower(), cpu_count=11, memory = 32)
    monster_executor = MonsterRemoteCommandLineCodeExecutor(client=client)
    ats = ATSTool(thread_id="tweet-sentiment-extraction")
    init_ats(ats)


    print("Your GPU Runtime is ready for action, Proceeding!")
    print(100*'#')
    try:
        print(f"Problem is:\n{message}")
        planner = InitialPlanner(problem_statement=message,executor=monster_executor)
        get_status = get_stage_status()
        print("*"*100)
        print(f"{get_status}")
    except KeyboardInterrupt as e:
        print("Exited..")
        # autogen.runtime_logging.stop()
