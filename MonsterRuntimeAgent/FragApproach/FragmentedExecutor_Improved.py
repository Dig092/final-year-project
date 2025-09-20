from MonsterRuntimeAgent.Tools.Gemini import GeminiContentGenerator
from MonsterRuntimeAgent.Tools.Anthropic import get_structured_response
from MonsterRuntimeAgent.FragApproach.MultiStageExecutor import DataEngineer, MachineLearningEngineer
from MonsterRuntimeAgent.Tools.ExperimentationModel import ExperimentPlanner
from MonsterRuntimeAgent.Tools.RuntimeTools import MonsterNeoCodeRuntimeClient
from MonsterRuntimeAgent.MonsterRuntimeCodeExecutor import MonsterRemoteCommandLineCodeExecutor

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from google.generativeai.types import GenerationConfig

EXAMPLE_INPUT = """
# Task

Detect apple diseases from images.

# Metric

Mean column-wise ROC AUC.

# Submission Format

For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

# Dataset

Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels
"""

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
    id: int = Field(description="Incremental integer experiment ID increment with each experiment.")
    experiment_name: str = Field(description="Short descriptive name for the experiment.")
    experiment_problem_statement: str = Field(description="Problem statement being tackled by this specific experiment.")
    success: bool = Field(description="True/False whether the iteration is successful or failed")
    experiment_outcome: str = Field(description="Detailed outcome of experiment if experiment has succeeded, if failed error and possible resolution.")

class ProblemStatement(BaseModel):
    experiment_name: str = Field(description="Short descriptive name for the experiment.")
    experiment_problem_statement: str = Field(description="Problem statement being tackled by this specific experiment.")

class EachPlan(BaseModel):
    plan_name: str = Field(description="Short descriptive plan name for the approach of experiment")
    plan_details: str = Field(description="Brief description of approach to follow")
    pros: str = Field(description="Brief bulleted pros")
    cons: str = Field(description="Brief bulleted cons")

class SuggestedPlans(BaseModel):
    plans_suggested: List[EachPlan]

class ExperimentOutcome(BaseModel):
    outcome: Literal["success", "failure"]
    outcome_reason: str = Field(description="Reason for outcome, tell what worked what did not if a error if causing a fail include what and a potential solution.")

class ExperimentFlow:
    def __init__(
        self, 
        problem_statement: str = EXAMPLE_INPUT,
        compute_info: str = "",
        max_experiments: int = 10,
        temperature: float = 0.7
    ):
        self.original_problem_statement = problem_statement
        self.compute_info = compute_info
        self.max_experiments = max_experiments
        self.experiments_list: List[ExperimentationRecord] = []
        
        # Initialize components
        self.plan = plan_better(self.original_problem_statement, compute_info=compute_info)
        self.plan_index = 0
        self.selected_plan = self.plan.plans_suggested[0]  # Start with first plan
        
        self.dataprep_obj = DataPrep(problem_statement=problem_statement)
        self.generator = GeminiContentGenerator(
            generation_config=GenerationConfig(
                temperature=temperature,
                top_p=0.8,
                top_k=40,
            )
        )
        
        # Initialize execution client
        self.client = MonsterNeoCodeRuntimeClient(container_type="gpu", cpu_count=8, memory=32)
        self.monster_executor = MonsterRemoteCommandLineCodeExecutor(client=self.client)

    def get_next_problem_statement(self) -> ProblemStatement:
        """
        Generate the next problem statement based on previous experiments and current context.
        """
        context = self._build_context()
        next_experiment = self._generate_next_experiment(context)
        self._validate_and_store_experiment(next_experiment)
        return next_experiment

    def _build_context(self) -> str:
        """Build context string for next experiment generation."""
        base_context = f"""
        Original Problem Statement:  
        {self.original_problem_statement}  

        Suggested Approach:
        {self.selected_plan}

        Data access instructions: 
        {self.dataprep_obj.data_journal}

        Node Compute Info:
        {self.compute_info}
        """

        if not self.experiments_list:
            return base_context + """
            This is the First experiment in approach:
            1. Focus on establishing working end to end pipeline from data loading to saving the model
            2. Use only a subset of data to establish the code
            3. Optimize for minibatches for maximum GPU utilization
            """
        
        last_experiment = self.experiments_list[-1]
        return base_context + f"""
        Previous Experiment Summary:
        - ID: {last_experiment.id}
        - Name: {last_experiment.experiment_name}
        - Success: {last_experiment.success}
        - Outcome: {last_experiment.experiment_outcome}

        Based on this outcome, suggest the next logical step in experimentation.
        """

    def _generate_next_experiment(self, context: str) -> ProblemStatement:
        """Generate next experiment based on context."""
        experiment = self.generator.generate_structured_content(context, ProblemStatement)
        
        # Update the problem statement with full context
        experiment.experiment_problem_statement = f"""
        {context}
        
        Current Problem to Solve:
        {experiment.experiment_problem_statement}
        """
        return experiment

    def _validate_and_store_experiment(self, experiment: ProblemStatement):
        """Validate and store the experiment record."""
        next_id = len(self.experiments_list) + 1
        record = ExperimentationRecord(
            id=next_id,
            experiment_name=experiment.experiment_name,
            experiment_problem_statement=experiment.experiment_problem_statement,
            success=False,
            experiment_outcome="Pending execution"
        )
        self.experiments_list.append(record)

    def perform_experiment(self, record: ExperimentationRecord) -> ExperimentationRecord:
        """Execute a single experiment and update its record."""
        try:
            engineer = MachineLearningEngineer(
                problem_statement=record.experiment_problem_statement,
                executor=self.monster_executor,
                rounds=30
            )
            engineering_summary = engineer.get_engineering_summary()
            
            # Analyze outcome
            outcome = self.analyze_outcome(engineering_summary)
            record.success = outcome.outcome == "success"
            record.experiment_outcome = outcome.outcome_reason
            
        except Exception as e:
            record.success = False
            record.experiment_outcome = f"Execution failed: Reason Unkown!"
            
        return record

    def analyze_outcome(self, engineering_summary: str) -> ExperimentOutcome:
        """Analyze the experiment outcome from engineering summary."""
        return self.generator.generate_structured_content(
            engineering_summary,
            ExperimentOutcome
        )

    def run_experiments(self) -> List[ExperimentationRecord]:
        """
        Run the complete experiment flow until success or max experiments reached.
        """
        while len(self.experiments_list) < self.max_experiments:
            # Get next problem statement
            problem_statement = self.get_next_problem_statement()
            
            # Execute experiment
            current_record = self.experiments_list[-1]
            updated_record = self.perform_experiment(current_record)
            
            # Update record
            self.experiments_list[-1] = updated_record
            
            # Check if we should continue
            if updated_record.success:
                break
            
            # If current plan failed multiple times, try next plan
            if (len(self.experiments_list) >= 3 and 
                not any(exp.success for exp in self.experiments_list[-3:])):
                self.plan_index += 1
                if self.plan_index < len(self.plan.plans_suggested):
                    self.selected_plan = self.plan.plans_suggested[self.plan_index]
                else:
                    break  # All plans exhausted
        
        return self.experiments_list

    def get_summary(self) -> dict:
        """Get a summary of all experiments."""
        return {
            "total_experiments": len(self.experiments_list),
            "successful_experiments": sum(1 for exp in self.experiments_list if exp.success),
            "current_plan": self.selected_plan.plan_name,
            "experiments": [exp.dict() for exp in self.experiments_list]
        }

# Helper function remains the same
def plan_better(problem_statement, model="claude-3-5-sonnet-20241022", temperature=0.2, compute_info=""):
    upgraded_prompt = f"""
    Problem Statement:
    {problem_statement}
    """
    return get_structured_response(
        text=upgraded_prompt,
        output_model=SuggestedPlans,
        system_prompt=f"""
        For given problem statement think as a ML Expert
        Can you give me multiple not more than 4 approaches I can follow?
        Order them based on best possible solution and gpu friendly faster convergence.
        consider this compute info to suggest better.
        {compute_info}
        Dont write code for me just plan
        """,
        model=model,
        temperature=temperature
    )


if __name__ == "__main__":
    compute_info = """
    10 cpus
    32 GB RAM
    A100-40GB GPU
    """
    import sys
    
    try:
        file_n = sys.argv[1]
    except IndexError:
        file_n = "facebook-recruiting-iii-keyword-extraction.md"

    path = f"/home/dev/MDockerRuntimeAPI/MonsterRuntimeAgent/competitions/{file_n}"
    problem_statement = open(path).read()
    eo = ExperimentFlow(problem_statement = problem_statement, compute_info = compute_info)