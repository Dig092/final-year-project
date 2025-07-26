import os
import json
import traceback
import google.generativeai as genai
from google.generativeai.types import ContentType, Tool, GenerationConfig
from typing import Dict, Any, Optional, Literal, TypeVar, Type, Callable, List
from pydantic import BaseModel, Field, ValidationError
import time
import inspect
from functools import wraps

T = TypeVar('T', bound=BaseModel)

class GeminiContentGenerator:
    """
    A class to generate content using the Gemini API with awareness of Pydantic model structures and function calling.
    """

    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model_name: Literal["gemini-1.5-pro", "gemini-1.5-flash"] = "gemini-1.5-flash",
                 generation_config: Optional[GenerationConfig] = None):
        """
        Initialize the GeminiContentGenerator.

        Args:
            api_key (Optional[str]): Your Google API key. If not provided, it will try to use the GEMINI_API_KEY environment variable.
            model_name (Literal["gemini-1.5-pro", "gemini-1.5-flash"]): The name of the Gemini model to use.
            generation_config (Optional[GenerationConfig]): Configuration for content generation.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY environment variable or pass api_key argument!")
        
        self.model_name = model_name
        self.generation_config = generation_config or GenerationConfig(temperature=0)
        genai.configure(api_key=self.api_key)
        
        self.registered_functions: Dict[str, Callable] = {}
        self.tools: List[Tool] = []
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the Gemini model with current configuration."""
        model_kwargs = {
            "model_name": self.model_name,
            "generation_config": self.generation_config
        }
        
        if self.tools:
            model_kwargs["tools"] = self.tools
            
        self.model = genai.GenerativeModel(**model_kwargs)

    def describe_pydantic_model(self, model: Type[BaseModel]) -> str:
        """
        Generate a description of the Pydantic model structure.

        Args:
            model (Type[BaseModel]): The Pydantic model to describe.

        Returns:
            str: A string description of the model structure.
        """
        schema = model.schema()
        description = f"Model Name: {schema['title']}\n\n"
        description += "Fields:\n"
        for field_name, field_info in schema['properties'].items():
            field_type = field_info.get('type', 'any')
            field_description = field_info.get('description', 'No description provided')
            description += f"- {field_name} ({field_type}): {field_description}\n"
        return description

    def generate_structured_content(self, prompt: str, model: Type[T], max_retries: int = 3, retry_delay: float = 1.0) -> T:
        """
        Generates structured content using the Gemini API and validates it against a Pydantic model.

        Args:
            prompt (str): The prompt to send to the Gemini API.
            model (Type[T]): The Pydantic model to validate against.
            max_retries (int): Maximum number of retries if validation fails.
            retry_delay (float): Delay in seconds between retries.

        Returns:
            T: An instance of the provided Pydantic model.

        Raises:
            ValueError: If unable to generate valid content after max_retries.
        """
        model_description = self.describe_pydantic_model(model)
        enhanced_prompt = f"""
        Please generate content based on the following prompt, ensuring it follows the structure defined below:

        {model_description}

        Prompt: {prompt}

        Provide the output as a valid JSON object that matches the described structure.
        """

        # Start a chat session for more coherent structured content generation
        chat = self.model.start_chat()

        for attempt in range(max_retries):
            try:
                # Send message through chat
                response = chat.send_message(enhanced_prompt)
                
                if not response.text:
                    raise ValueError("Empty response from API")

                # Try to parse as JSON
                try:
                    json_content = json.loads(response.text)
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract JSON from the text
                    json_start = response.text.find('{')
                    json_end = response.text.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_content = json.loads(response.text[json_start:json_end])
                    else:
                        # If JSON extraction fails, ask for a fix
                        fix_prompt = "The previous response was not in valid JSON format. Please provide the response in strict JSON format only."
                        response = chat.send_message(fix_prompt)
                        if not response.text:
                            raise ValueError("Empty response from API during JSON fix attempt")
                        json_content = json.loads(response.text)

                # Validate against the Pydantic model
                validated_content = model.parse_obj(json_content)
                return validated_content

            except (ValueError, ValidationError) as e:
                if attempt < max_retries - 1:
                    error_message = f"Attempt {attempt + 1} failed with error: {str(e)}\nPlease fix the errors and ensure the response matches the required structure exactly."
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Send error feedback to the chat for better next attempt
                    chat.send_message(error_message)
                else:
                    raise ValueError(f"Failed to generate valid content after {max_retries} attempts. Last error: {str(e)}")

    def generate_content(self, prompt: str) -> Dict[str, Any]:
        """
        Generates content using the Gemini API based on the given prompt.

        Args:
            prompt (str): The prompt to send to the Gemini API.

        Returns:
            Dict[str, Any]: A dictionary containing the generated content and metadata.
        """
        try:
            response = self.model.generate_content(prompt)
            
            result = {
                "content": response.text,
                "metadata": {
                    "model": self.model_name,
                    "prompt_feedback": response.prompt_feedback
                }
            }
        except Exception as e:
            result = {
                "error": str(e),
                "content": None,
                "metadata": None
            }
            traceback.print_exc()
        
        return result

# Example usage and test code
if __name__ == "__main__":
    print("\n=== Example 1: Basic Content Generation ===\n")
    
    # Create generator with custom generation config
    generator = GeminiContentGenerator(
        generation_config=GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
        )
    )

    # Test basic content generation
    prompt = "Write a short poem about artificial intelligence."
    result = generator.generate_content(prompt)
    print("Basic Content Generation Result:")
    print(result)

    print("\n=== Example 2: Structured Content Generation ===\n")

    # Define a Pydantic model for experiment description
    class ExperimentDescription(BaseModel):
        title: str = Field(..., description="The title of the experiment")
        objective: str = Field(..., description="The main goal of the experiment")
        hypothesis: str = Field(..., description="The proposed explanation or prediction")
        methodology: Dict[str, str] = Field(..., description="Steps to conduct the experiment")
        expected_outcome: str = Field(..., description="Anticipated results of the experiment")
        computational_requirements: Dict[str, Any] = Field(..., description="Computational resources needed")
        optimization_metric: str = Field(..., description="The metric to be optimized in the experiment")

    # Generate structured content
    prompts = [
        "Generate an experiment description for a machine learning task to predict stock market trends.",
        "Create an experiment description for a natural language processing task to detect fake news."
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        try:
            experiment = generator.generate_structured_content(prompt, ExperimentDescription)
            experiment_json = experiment.model_dump_json()
            print("Generated Experiment Description:")
            print(json.dumps(json.loads(experiment_json)))
        except ValueError as e:
            print(f"Error: {e}")