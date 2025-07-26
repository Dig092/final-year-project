from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


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

    class Config:
        validate_assignment = True

class ExperimentPlanner:
    def __init__(self):
        """Initialize the experiment planner."""
        pass

    def plan_from_prompt(self, prompt: str) -> ExperimentPlan:
        """
        Generate compute requirements plan from user prompt.
        
        Args:
            prompt (str): User's experiment description
            
        Returns:
            ExperimentPlan: Compute and optimization requirements
        """
        # This is where you'd integrate with Gemini or other LLM
        # For now, returning an example plan
        return ExperimentPlan(
            target="customer_churn",
            user_input=prompt,
            problem_type=ProblemType.CLASSIFICATION,
            node_type=NodeType.GPU,
            gpu_size_gb=16,
            gpu_count=1,
            cpu_memory_gb=32,
            optimization_metrics=["accuracy", "recall"],
            optimization_targets={"accuracy": 0.85, "recall": 0.90}
        )

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
            planner.display_plan(plan)
        except Exception as e:
            print(f"Error planning experiment {i}: {e}")
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()