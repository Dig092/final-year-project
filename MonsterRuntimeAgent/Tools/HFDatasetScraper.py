import os
import requests
import pandas as pd
from datasets import load_dataset
from autogen import AssistantAgent, UserProxyAgent, register_function, config_list_from_json

# Load GPT-4 configuration from a JSON list
config_list = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model": ["gpt-4o"]}
)

llm_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}


class DatasetExpertAgent:
    """
    A class to act as a dataset expert, which can search, parse, filter, and gather detailed information
    about Hugging Face datasets.
    """

    def __init__(self, hf_token=None):
        """
        Initialize the agent with an optional Hugging Face token for authenticated requests.

        Args:
            hf_token (str): Hugging Face token for authenticated requests (optional).
        """
        self.hf_token = hf_token
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}

    def search_datasets(self, keyword: str, top_n: int = 5) -> list:
        """
        Search Hugging Face datasets using a keyword and retrieve the top N datasets.

        Args:
            keyword (str): The keyword to search datasets.
            top_n (int): Number of top datasets to retrieve.

        Returns:
            list: A list of dictionaries with dataset metadata.
        """
        search_url = "https://huggingface.co/api/datasets"
        params = {"search": keyword, "limit": top_n}
        response = requests.get(search_url, params=params, headers=self.headers)

        if response.status_code != 200:
            return f"Failed to fetch datasets. Status code: {response.status_code}"

        datasets = response.json()

        if not datasets:
            return f"No datasets found for the keyword: {keyword}"

        dataset_info = []
        for dataset in datasets:
            dataset_name = dataset['id']
            try:
                dataset_metadata = self.fetch_dataset_info(dataset_name)
                dataset_info.append({
                    'Dataset Name': dataset_name,
                    'Description': dataset.get('description', 'No description'),
                    'Size (MB)': dataset_metadata['size_mb'],
                    'Splits': dataset_metadata['splits'],
                    'Row Count': dataset_metadata['row_count'],
                    'Columns': dataset_metadata['columns'],
                    'Sample Rows': dataset_metadata['sample_rows']
                })
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")

        return dataset_info

    def fetch_dataset_info(self, dataset_name: str) -> dict:
        """
        Fetch additional metadata for a dataset, including size, column names, sample rows, and splits.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            dict: A dictionary containing metadata including size, columns, splits, and a few rows.
        """
        config_url = f"https://huggingface.co/api/datasets/{dataset_name}"
        config_response = requests.get(config_url, headers=self.headers)

        if config_response.status_code != 200:
            raise Exception(f"Failed to fetch dataset configuration for {dataset_name}")
        
        config_data = config_response.json()

        size_mb = config_data.get('size', 0) / (1024 ** 2)  # Convert to MB
        splits = config_data.get('splits', {})

        columns = self.get_dataset_columns(dataset_name)
        sample_rows = self.get_sample_rows(dataset_name)

        row_count = {split: split_info.get("num_rows", 0) for split, split_info in splits.items()}

        return {
            'size_mb': size_mb,
            'splits': list(splits.keys()),
            'row_count': row_count,
            'columns': columns,
            'sample_rows': sample_rows
        }

    def get_dataset_columns(self, dataset_name: str) -> list:
        """
        Fetch dataset column names.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            list: A list of column names.
        """
        dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)
        return list(dataset.column_names)

    def get_sample_rows(self, dataset_name: str, num_rows: int = 3) -> list:
        """
        Fetch a few sample rows from the dataset without loading the entire dataset.

        Args:
            dataset_name (str): The name of the dataset.
            num_rows (int): The number of rows to fetch.

        Returns:
            list: A list containing the sample rows.
        """
        try:
            dataset = load_dataset(dataset_name, split='train', trust_remote_code="true")
            sample_rows = dataset.select(range(num_rows))
            return sample_rows.to_pandas().to_dict(orient="records")
        except Exception as e:
            print(f"Error fetching sample rows for {dataset_name}: {e}")
            return []

    def check_columns(self, dataset_name: str, required_columns: list) -> str:
        """
        Check if the required columns are present in the dataset.

        Args:
            dataset_name (str): The name of the dataset.
            required_columns (list): A list of columns that must be present.

        Returns:
            str: A message indicating whether the columns are present or missing.
        """
        columns = self.get_dataset_columns(dataset_name)
        missing_columns = [col for col in required_columns if col not in columns]
        return "All required columns are present." if not missing_columns else f"Missing columns: {', '.join(missing_columns)}"

    def get_available_splits(self, dataset_name: str) -> list:
        """
        Retrieve available dataset splits.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            list: A list of available dataset splits.
        """
        splits_url = f"https://datasets-server.huggingface.co/splits?dataset={dataset_name}"
        response = requests.get(splits_url, headers=self.headers)

        if response.status_code != 200:
            print(f"Failed to fetch dataset splits. Status code: {response.status_code}")
            return []

        data = response.json()
        unique_subsets = list(set([obj["config"] for obj in data["splits"]]))
        return unique_subsets

    def get_summary(self, keyword: str, top_n: int = 5) -> str:
        """
        Get a summary of the top N datasets based on a keyword, including columns, splits, and sample rows.

        Args:
            keyword (str): Keyword to search for datasets.
            top_n (int): Number of datasets to retrieve.

        Returns:
            str: A summarized result of the top N datasets.
        """
        summary = f"Summary for keyword '{keyword}':\n\n"
        search_result = self.search_datasets(keyword, top_n)
        summary += f"Search Results:\n{pd.DataFrame(search_result).to_string()}\n\n"
        
        for dataset in search_result:
            dataset_name = dataset["Dataset Name"]
            columns = self.get_dataset_columns(dataset_name)
            sample_rows = self.get_sample_rows(dataset_name)
            splits = self.get_available_splits(dataset_name)
            summary += f"Dataset: {dataset_name}\nColumns: {columns}\nSplits: {splits}\nSample Rows: {sample_rows}\n\n"
        
        return summary


dataset_agent = DatasetExpertAgent()

# Example usage of the DatasetExpertAgent class
if __name__ == "__main__":
    # Initialize DatasetExpertAgent

    # Example: Search for movie-related datasets and get a summary
    summary = dataset_agent.get_summary("movies", top_n=3)
    print(summary)

# Step 2: Define the registerable functions (tools)
def search_datasets_tool(keyword: str, top_n: int = 5, dataset_agent = dataset_agent) -> str:
    return dataset_agent.search_datasets(keyword, top_n)

def get_summary_tool(keyword: str, top_n: int = 3, dataset_agent = dataset_agent) -> str:
    return dataset_agent.get_summary(keyword, top_n)

"""
# Step 3: Initialize AutoGen Assistant and Register the Tools
assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy")

# Register the tools with the assistant
register_function(search_datasets_tool, caller=assistant, executor=user_proxy, name="search_datasets", description="Search for datasets.")
register_function(get_summary_tool, caller=assistant, executor=user_proxy, name="get_summary", description="Get a search summary of datasets.")

# Step 4: Initiate a chat example
user_proxy.initiate_chat(
    assistant,
    message="Search for movie datasets and get a summary of the top 10 results."
)
"""
