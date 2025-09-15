from anthropic import Anthropic
from typing import TypeVar, Type, Optional
from pydantic import BaseModel
import json
import os

T = TypeVar('T', bound=BaseModel)

def get_structured_response(
    text: str,
    output_model: Type[T],
    api_key: str = os.environ.get("ANTHROPIC_API_KEY"),
    model: str = "claude-3-haiku-20240307",
    max_tokens: int = 1024,
    temperature: float = 0.2,
    system_prompt: Optional[str] = None
) -> T:
    """
    Get a structured response from Claude API following a Pydantic model schema.
    
    Args:
        text (str): Input text to analyze
        output_model (Type[T]): Pydantic model class to structure the output
        api_key (str): Anthropic API key
        model (str): Claude model to use
        max_tokens (int): Maximum tokens in response
        temperature (float): Temperature for response generation
        system_prompt (Optional[str]): Custom system prompt. If None, uses default.
    
    Returns:
        T: Instance of the provided Pydantic model
    
    Raises:
        ValueError: If the API response cannot be parsed into the model
        HTTPError: If the API request fails
    """
    
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Get the JSON schema from the Pydantic model
    model_schema = output_model.model_json_schema()
    
    # Define tools for structured output
    tools = [{
        "name": "build_structured_output",
        "description": f"Build a structured output following the schema",
        "input_schema": model_schema
    }]
    
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = (
            "Analyze the provided text and return a structured output following "
            "the exact schema provided. Be precise and comprehensive in the analysis. "
            "Ensure all required fields are filled and follow any constraints in the schema."
        )
    
    try:
        # Make API call
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"{text}"
                }
            ],
            tools=tools,
            tool_choice={"type": "tool", "name": "build_structured_output"}
        )
        
        # Extract the function call input
        if not response.content or not response.content[0].input:
            raise ValueError("No structured output received from Claude")
            
        function_call = response.content[0].input
        
        # Parse the response into the Pydantic model
        try:
            return output_model(**function_call)
        except Exception as e:
            raise ValueError(f"Failed to parse Claude's response into the model: {str(e)}")
            
    except Exception as e:
        raise Exception(f"Error calling Claude API: {str(e)}")

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

# Simple sentiment enum
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

# Simple aspect analysis model
class AspectAnalysis(BaseModel):
    aspect: str
    sentiment: Sentiment
    description: str

# Main analysis model
class SimpleReviewAnalysis(BaseModel):
    overall_sentiment: Sentiment
    summary: str
    aspects: List[AspectAnalysis]
    pros: List[str]
    cons: List[str]

def test_review_analysis():
    # Sample reviews to test
    reviews = [
        """
        This camera is fantastic! The image quality is crystal clear and the battery 
        lasts all day. The only downside is that it's a bit heavy and the menu system 
        is confusing at first. But overall, I'm very happy with my purchase.
        """,
        
        """
        The restaurant was okay. Food came quickly and was hot, but the taste was 
        average. The service was friendly though the waiter forgot our drinks. 
        Prices were reasonable but I probably won't go back.
        """
    ]

    for i, review in enumerate(reviews, 1):
        print(f"\n=== Testing Review #{i} ===")
        try:
            result = get_structured_response(
                text=review,
                output_model=SimpleReviewAnalysis,
                system_prompt="""
                Analyze this review and provide:
                1. Overall sentiment
                2. A brief summary
                3. Analysis of different aspects mentioned
                4. Clear pros and cons
                
                Be specific and concise in your analysis.
                """,
                temperature=0.1
            )
            
            # Print results in a readable format
            print("\nOverall Sentiment:", result.overall_sentiment)
            print("\nSummary:", result.summary)
            
            print("\nAspect Analysis:")
            for aspect in result.aspects:
                print(f"- {aspect.aspect}: {aspect.sentiment}")
                print(f"  {aspect.description}")
            
            print("\nPros:")
            for pro in result.pros:
                print(f"- {pro}")
            
            print("\nCons:")
            for con in result.cons:
                print(f"- {con}")

        except Exception as e:
            print(f"Error analyzing review: {str(e)}")

# Example expected output structure:
"""
{
    "overall_sentiment": "positive",
    "summary": "Positive review of a camera with great image quality and battery life, despite some usability issues.",
    "aspects": [
        {
            "aspect": "image quality",
            "sentiment": "positive",
            "description": "Crystal clear images noted as a major strength"
        },
        {
            "aspect": "battery life",
            "sentiment": "positive",
            "description": "Lasts all day, meeting user needs"
        },
        {
            "aspect": "usability",
            "sentiment": "negative",
            "description": "Menu system described as confusing for new users"
        }
    ],
    "pros": [
        "Excellent image quality",
        "Long battery life"
    ],
    "cons": [
        "Heavy weight",
        "Confusing menu system"
    ]
}
"""

if __name__ == "__main__":
    test_review_analysis()