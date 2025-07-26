from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from google.generativeai.types import GenerationConfig

from MonsterRuntimeAgent.Tools.Gemini import GeminiContentGenerator
from MonsterRuntimeAgent.Tools.NetScraper import retreive_from_internet


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    NLP = "nlp"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    OTHER = "other"

class NodeType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"

class ExperimentPlan(BaseModel):
    """Simplified experiment planning structure focused on compute requirements."""
    
    # Basic experiment info
    target: str = Field(..., description="What the experiment aims to predict/detect/solve")
    user_input: str = Field(..., description="Original user provided input text")
    problem_type: ProblemType = Field(..., description="Type of ML/AI problem")
    optimization_metric: List[str] = Field(..., description= "List of metrics to optimize as part of problem, like RMSE, Jaccard Score e.t.c!")
    optimization_target: Dict[str, float] = Field(..., description = "Decided metrics and modest target values to hit, if user specified something use them!")
    
    # Compute requirements
    node_type: NodeType = Field(
        NodeType.GPU,
        description="Primary compute type needed (CPU/GPU)"
    )
    
    # GPU specific estimates (if applicable)
    gpu_size_gb: Optional[int] = Field(
        None,
        description="Estimated GPU memory needed in GB",
        ge=8,
        le=80
    )
    gpu_count: Optional[int] = Field(
        None,
        description="Number of GPUs needed",
        ge=1,
        le=8
    )
    
    # CPU specific estimates (if applicable)
    cpu_count: Optional[int] = Field(
        None,
        description="Number of CPU cores needed",
        ge=1
    )
    cpu_memory_gb: Optional[int] = Field(
        None,
        description="RAM needed in GB",
        ge=4
    )
    
    # Optimization targets
    optimization_metrics: List[str] = Field(
        default=[],
        description="Metrics to optimize for (e.g. accuracy, f1)"
    )
    optimization_targets: Optional[dict] = Field(
        None,
        description="Target values for optimization metrics"
    )

    key_additional_notes: Optional[str] = Field(None, description="Use this field to store important key information provided in prompt \
                                                from background research and e.t.c! Use this to seed/store required additional information \
                                                to solve the problem statement. Additional dataset links.")

    class Config:
        validate_assignment = True

class ExperimentPlanner:
    def __init__(self):
        """Initialize the experiment planner."""
        self.generator = GeminiContentGenerator(
            generation_config=GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
            )
        )

    def plan_from_prompt(self, prompt: str) -> ExperimentPlan:
        """
        Generate compute requirements plan from user prompt.
        
        Args:
            prompt (str): User's experiment description
            
        Returns:
            ExperimentPlan: Compute and optimization requirements
        """
        background_research = self.perform_background_research(prompt)

        formatted_prompt = f"""
        user_prompt: 
        {prompt}
        
        background information:
        {background_research}
        """
        experiment = self.generator.generate_structured_content(formatted_prompt, ExperimentPlan)
        return experiment

    def display_plan(self, plan: ExperimentPlan) -> None:
        """Display the experiment plan in a readable format."""
        print("\n=== Experiment Plan ===")
        print(f"Target: {plan.target}")
        print(f"Problem Type: {plan.problem_type.value}")
        print("\nCompute Requirements:")
        print(f"Node Type: {plan.node_type.value.upper()}")
        
        if plan.node_type == NodeType.GPU:
            print(f"GPU Memory: {plan.gpu_size_gb}GB")
            print(f"GPU Count: {plan.gpu_count}")
        
        print(f"CPU Memory: {plan.cpu_memory_gb}GB")
        
        if plan.optimization_metrics:
            print("\nOptimization Goals:")
            for metric in plan.optimization_metrics:
                target = plan.optimization_targets.get(metric) if plan.optimization_targets else None
                print(f"- {metric}: {target if target else 'Not specified'}")

        print("key Additional Notes!")
        print(100*('-'))
        print(plan.key_additional_notes)
        print(100*('-'))

    def perform_background_research(self, prompt):
        return retreive_from_internet(prompt)

    def tree_of_thoughts_plan(self):
        pass

def main():
    """Example usage of the experiment planner."""
    planner = ExperimentPlanner()
    
    prompts = [
        """
        Build a customer churn prediction model using historical data.
        Need at least 85% accuracy and 90% recall.
        Data includes usage patterns, billing info, and support tickets.
        """,
        """
        Create an object detection model for finding defects in circuit boards.
        Using 4K camera images, need real-time processing.
        Target 95% detection rate with low false positives.
        """
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing experiment {i}...")
        try:
            plan = planner.plan_from_prompt(prompt)
            import pdb;pdb.set_trace()
            planner.display_plan(plan)
        except Exception as e:
            print(f"Error planning experiment {i}: {e}")
        finally:
            break
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()