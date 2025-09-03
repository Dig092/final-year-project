
from MonsterRuntimeAgent.Tools.Gemini import GeminiContentGenerator
from MonsterRuntimeAgent.Tools.MultiStageExecutor import DataEngineer
from MonsterRuntimeAgent.MonsterRuntimeCodeExecutor import MonsterRemoteCommandLineCodeExecutor
from MonsterRuntimeAgent.Tools.RuntimeTools import MonsterNeoCodeRuntimeClient

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
        Involing only download, minimal transformation.

        Make sure to perform any transformation inline with python execution.

        Make sure to include download instruction if you know any.
        """
        experiment = self.generator.generate_structured_content(updated_prompt, DataPrepModel)
        experiment_json = experiment.model_dump_json()
        return experiment_json

    def perform_groupchat_to_solve_data_prep(self):
        data_engineer = DataEngineer(problem_statement=self.data_prep_problem_statement,executor=self.monster_executor)
        return data_engineer.get_planner_summary()



if __name__ == "__main__":
    dp = DataPrep()
    import pdb;pdb.set_trace()
