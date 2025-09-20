from MonsterRuntimeAgent.Tools.Gemini import GeminiContentGenerator
from MonsterRuntimeAgent.Tools.Anthropic import get_structured_response
from MonsterRuntimeAgent.FragApproach.MultiStageExecutor import DataEngineer, MachineLearningEngineer
from MonsterRuntimeAgent.Tools.ExperimentationModel import ExperimentPlanner
from MonsterRuntimeAgent.Tools.RuntimeTools import MonsterNeoCodeRuntimeClient
from MonsterRuntimeAgent.MonsterRuntimeCodeExecutor import MonsterRemoteCommandLineCodeExecutor

import anthropic

from typing import List

EXAMPLE_INPUT = """
# Task
Extract the part of the tweet that reflects the sentiment (positive, negative, or neutral) given by the provided sentiment label.

# Evaluation Metric
The evaluation metric for the competition is the word-level Jaccard score.

# Submission Format
- A CSV file with the following columns:
  - `textID`: Unique ID for the tweet
  - `selected_text`: The portion of the tweet that reflects the sentiment

Example:
textID,selected_text
f941f4d7fa,"what a great day"
f941f4d7fa,"it was ok"


# Dataset
The dataset consists of tweets along with sentiment labels.

### Dataset Structure:
- **twitter_training.csv**: Contains the training dataset with text, sentiment labels, and selected text.
- **twitter_validation.csv**: Contains the test dataset with text and sentiment labels.
- **sample_submission.csv**: A sample format of the submission file.

### Columns in the Dataset:
- **textID**: Unique identifier for each tweet.
- **text**: The tweet itself.
- **sentiment**: Sentiment label of the tweet, which can be positive, negative, or neutral.
- **selected_text**: (in train.csv only) The portion of the tweet that reflects the sentiment.

### Dataset Name:
- Dataset can be download from kaggle using `jp797498e/twitter-entity-sentiment-analysis`

Assume runtime already has kaggle API Auth configured.

Top score is 0.74 in score try to beat it!

"""

EXAMPLE_INPUT = """
# Task

Categorise plant images.

# Metric

Macro F1 score. In "macro" F1 a separate F1 score is calculated for each `species` value and then averaged. 

# Submission Format 

For each image `Id`, you should predict the corresponding image label (`category_id`) in the `Predicted` column. The submission file should have the following format:
```
Id,Predicted
0,1
1,27
2,42
...
```

# Dataset

In addition to the images, we also include a hierarchical taxonomic structure of `category_id`. The `categories` in the `training_metadata.json` contain three levels of hierarchical structure, `family` - `genus` - `species`, from the highest rank to the lowest rank. One can think about this as a directed graph, where families are the root nodes and the species are the leaf nodes. Please note that `species` is only unique under its parent node `genus`, which means that we can find multiple categories with the same `species` name under different `genus` names. This is due to the taxonomic nature that plants are named after, and the `genus`-`species` pair is always unique in our data.

Pairwise phylogenetic distances among genera is included in key name `distances` from `train_metadata.json`.

The training set images are organized in subfolders `h22-train/images/<subfolder1>/<subfolder2>/<image_id>.jpg`, where `<subfolder1>` and `<subfolder2>` comes from the first three and the last two digits of the image_id. Image_id is a result of combination between `<category_id>` and unique numbers that differentiates images within plant taxa. Please be mindful that `category_id`s are unique, but not complete. Taxa are originally numbered from 1 to 15505, but the competition data has 15501 taxa because we lost four taxa during data cleaning process.

The test set images are organized in subfolders `test/images/<subfolder>/<image id>.jpg`, where `<subfolder>` corresponds to the integer division of the `image_id` by 1000. For example, a test image with and `image_id` of `8005`, can be found at `h22-test/images/008/test-008005.jpg`

kaggle competitions download -c herbarium-2022-fgvc9 # use kaggle api to download from inside python

create a test split from train with 10% use random_state of 0

kaggle competitions download -c herbarium-2022-fgvc9

Use train data to create per category 20% into test/validation set.
"""

from pydantic import BaseModel, Field
from google.generativeai.types import GenerationConfig

class DataPrepModel(BaseModel):
    problem_statement: str = Field(description = "Data Preparation problem statement detail include suggested code steps dataset name and all required info to download required data.")

class DataPrep():
    def __init__(self, problem_statement: str = EXAMPLE_INPUT):
        self.problem_statement = problem_statement
        self.generator = GeminiContentGenerator(
        generation_config=GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            )
        )
        self.data_prep_problem_statement = self.extract_dataprep_problem_statement()
        self.client = MonsterNeoCodeRuntimeClient(container_type="gpu", cpu_count=8, memory = 16)
        self.monster_executor = MonsterRemoteCommandLineCodeExecutor(client=self.client)
        self.data_journal = self.perform_groupchat_to_solve_data_prep()

    def extract_dataprep_problem_statement(self):
        updated_prompt = f"""
        Original Problem: {self.problem_statement}

        Considering above problem generate a problem statement to perform required intial data prep.
        Involving only download, minimal transformation.

        Make sure to perform any transformation inline with python execution.

        Make sure to include download instruction if you know any. 

        Also include about dataset information like size, type e.t.c 
        """
        experiment = self.generator.generate_structured_content(updated_prompt, DataPrepModel)
        experiment_json = experiment.model_dump_json()
        return experiment_json

    def perform_groupchat_to_solve_data_prep(self):
        data_engineer = DataEngineer(problem_statement=self.data_prep_problem_statement,executor=self.monster_executor)
        return data_engineer.get_planner_summary()

class ExperimentationRecord(BaseModel):
    id: int = Field(description = "Incremental integer experiment ID increment with each experiment.")
    experiment_name: str = Field(description = "Short descriptive name for the experiment.")
    experiment_problem_statement: str = Field(description = "Problem statement being tackled by this specific experiment.")
    success: bool = Field(description = "True/False whether the iteration is successfull of failed")
    experiment_outcome: str = Field(description = "Detailed outcome of experiment if experiment has succeeded, if failed error and possible resolution.")

class ProblemStatement(BaseModel):
    experiment_name: str = Field(description = "Short descriptive name for the experiment.")
    experiment_problem_statement: str = Field(description = "Problem statement being tackled by this specific experiment.")

class EachPlan(BaseModel):
    plan_name: str = Field(description = "Short descriptive plan name for the approach of experiment")
    plan_details: str = Field(description = "Brief description of approach to follow")
    pros: str = Field(description = "Brief bulleted pros")
    cons: str = Field(description = "Brief bulleted cons")

class SuggestedPlans(BaseModel):
    plans_suggested: List[EachPlan]

def plan_better(problem_statement, model = "claude-3-5-sonnet-20241022", temperature = 0.2, compute_info = ""):
    upgraded_prompt = f"""
    Problem Statement:
    {problem_statement}
    """
    return get_structured_response(
                text=upgraded_prompt,
                output_model=SuggestedPlans,
                system_prompt= f"""
                For given problem statement think as a ML Expert
                Can you give me multiple not more than 4 approaches I can follow ?
                Order them based on best possible solution and gpu friendly faster convergence.
                consider this compute info to suggest better.
                {compute_info}
                Dont write code for me just plan
                """,
                model = model,  
                temperature=temperature
            )


class ExperimentFlow:
    def __init__(self, problem_statement: str = EXAMPLE_INPUT, compute_info: str = ""):
        self.original_problem_statement = problem_statement
        self.compute_info = compute_info
        self.plan = plan_better(self.original_problem_statement, compute_info = compute_info)
        self.plan_index = 0 
        self.select_best_plan()
        self.dataprep_obj = DataPrep(problem_statement=problem_statement)
        self.generator = GeminiContentGenerator(
        generation_config=GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            )
        )
        self.experiments_list = []
    
    def select_best_plan(self):
        self.selected_plan = self.plan.plans_suggested[self.plan_index]
        self.plan_index += self.plan_index
    
    def create_tot_problem_statement(self) -> str:
        planner =  ExperimentPlanner()
        tot_plan = planner.tree_of_thoughts_plan(problem=self.original_problem_statement)
        return tot_plan

    def create_experiment_record_and_append_to_list(self, problem_statement_model: ProblemStatement):
        """
        Convert a ProblemStatement model to an ExperimentationRecord and append it to experiments_list.
        
        Args:
            problem_statement_model (ProblemStatement): The problem statement model to convert
            
        Returns:
            None: Updates self.experiments_list in place
        """
        # Get the next ID based on the last experiment in the list
        next_id = 1
        if self.experiments_list:
            next_id = self.experiments_list[-1].id + 1
        
        # Create new ExperimentationRecord
        experiment_record = ExperimentationRecord(
            id=next_id,
            experiment_name=problem_statement_model.experiment_name,
            experiment_problem_statement=problem_statement_model.experiment_problem_statement,
            success=False,  # Initialize as False since experiment hasn't run yet
            experiment_outcome="Experiment not yet executed"  # Initial status
        )
        
        # Append to experiments list
        self.experiments_list.append(experiment_record)

    def get_next_problem_statement(self):
        if len(self.experiments_list) == 0:
            last_experiment_info = f"""
            Original Problem Statement:  
            {self.original_problem_statement}  

            Suggested Approach:
            {self.selected_plan}

            Data access instructions: 
            {self.dataprep_obj.data_journal}

            Node Compute Info:
            {self.compute_info}
    
            This is the First experiment in approach focus on establishing working end to end pipeline from data loading to saving the model.
            Use only a subset of data to establish the code.
            Optimize for minibatches for maximum GPU utilization. 
            """
        else:
            last_experiment = self.experiments_list[-1]
            last_experiment_info = f"""
            Original Problem Statement: 
            {self.original_problem_statement}

            Suggested Approach:
            {self.selected_plan}

            Data access instructions: 
            {self.dataprep_obj.data_journal}

            Node Compute Info:
            {compute_info}

            Previous Experiment Performed:
            id:{ExperimentationRecord.id}

            experiment_problem_statement: {ExperimentationRecord.experiment_problem_statement}
            
            success/Failure: {ExperimentationRecord.sucess}

            experiment_outcome: {ExperimentationRecord.experiment_outcome}

            Assume dataprep is already handled

            consider this info and suggest a proper next step complete problem statement encompassing all required information for the agent following to execute.
            """

        experiment = self.generator.generate_structured_content(last_experiment_info, ProblemStatement)

        updated_prompt = f"""
        Original Problem Statement: 
        {self.original_problem_statement}

        Suggested Approach:
        {self.selected_plan}

        Data access instructions: 
        {self.dataprep_obj.data_journal}

        Node Compute Info:
        {compute_info}

        Current Problem to Solve:
        {experiment.experiment_problem_statement}
        """
        experiment.experiment_problem_statement = updated_prompt
        self.create_experiment_record_and_append_to_list(experiment)
        return experiment




if __name__ == "__main__":
    #dp = DataPrep()
    compute_info = """
    10 cpus
    32 GB RAM
    A100-40GB GPU
    """
    client = MonsterNeoCodeRuntimeClient(container_type="gpu", cpu_count=8, memory = 32)
    monster_executor = MonsterRemoteCommandLineCodeExecutor(client=client)
    ef = ExperimentFlow(compute_info = compute_info)
    problem_statement = ef.get_next_problem_statement()
    mle_obj = MachineLearningEngineer(problem_statement = problem_statement.experiment_problem_statement, executor=monster_executor, rounds = 30)
    engineering_summary = mle_obj.get_engineering_summary()
    import pdb;pdb.set_trace()
