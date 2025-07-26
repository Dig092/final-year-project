from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import uuid
import logging
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentStatus(str, Enum):
    PLANNED = "planned"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

class ExperimentType(str, Enum):
    ML = "machine_learning"
    DATA_ANALYSIS = "data_analysis"
    AB_TEST = "ab_testing"
    OTHER = "other"

class ExperimentResult(BaseModel):
    """Final experiment results."""
    metrics: Dict[str, float] = Field(default_factory=dict)
    conclusions: Optional[str] = None
    completed_at: datetime = Field(default_factory=datetime.now)

class Experiment(BaseModel):
    """Simplified experiment tracking model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    title: str
    description: str
    type: ExperimentType
    status: ExperimentStatus = ExperimentStatus.PLANNED
    
    # Configuration and parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Planning metadata
    created_at: datetime = Field(default_factory=datetime.now)
    target_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Final results (updated once at completion)
    result: Optional[ExperimentResult] = None
    
    # Documentation
    notes: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True

class ExperimentTracker:
    """Simple experiment planning and tracking system."""
    
    def __init__(self, storage_dir: Union[str, Path]):
        self.storage_dir = Path(storage_dir)
        self.experiments_dir = self.storage_dir / "experiments"
        self.graph_path = self.storage_dir / "experiment_graph.json"
        self.chromadb_path = self.storage_dir / "chromadb"
        
        # Create directories
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.chromadb_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize graph
        self.graph = nx.DiGraph()
        self._load_graph()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chromadb_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.experiment_collection = self.chroma_client.get_or_create_collection(
            name="experiments",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        
        logger.info(f"Initialized ExperimentTracker at {storage_dir}")

    def plan_experiment(
        self,
        title: str,
        description: str,
        exp_type: ExperimentType,
        parent_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        target_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None
    ) -> Experiment:
        """Plan a new experiment."""
        experiment = Experiment(
            title=title,
            description=description,
            type=exp_type,
            parent_id=parent_id,
            parameters=parameters or {},
            target_metrics=target_metrics or {},
            tags=tags or []
        )
        
        if parent_id:
            parent = self.get_experiment(parent_id)
            if parent:
                experiment.parameters = {**parent.parameters, **experiment.parameters}
                experiment.tags = list(set(parent.tags + experiment.tags))
        
        self._save_experiment(experiment)
        self._add_to_graph(experiment)
        self._update_chromadb(experiment)
        
        logger.info(f"Planned new experiment: {experiment.id} - {experiment.title}")
        return experiment

    def complete_experiment(
        self,
        exp_id: str,
        metrics: Dict[str, float],
        conclusions: Optional[str] = None,
        status: ExperimentStatus = ExperimentStatus.COMPLETED
    ) -> Experiment:
        """Mark an experiment as completed with its final results."""
        experiment = self.get_experiment(exp_id)
        if not experiment:
            raise ValueError(f"Experiment {exp_id} not found")

        experiment.status = status
        if status == ExperimentStatus.COMPLETED:
            experiment.result = ExperimentResult(
                metrics=metrics,
                conclusions=conclusions,
                completed_at=datetime.now()
            )

        self._save_experiment(experiment)
        self._update_chromadb(experiment)
        
        logger.info(f"Completed experiment: {experiment.id}")
        return experiment

    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        """Retrieve an experiment by ID."""
        exp_path = self.experiments_dir / f"{exp_id}.json"
        if not exp_path.exists():
            return None
        
        with open(exp_path, 'r') as f:
            data = json.load(f)
        return Experiment(**data)

    def search_experiments(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Experiment]:
        """Search experiments using semantic similarity."""
        results = self.experiment_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )
        
        experiments = []
        if results and results['ids']:
            for exp_id in results['ids'][0]:
                experiment = self.get_experiment(exp_id)
                if experiment:
                    experiments.append(experiment)
        
        return experiments

    def visualize_experiments(
        self,
        output_path: Optional[Path] = None,
        highlight_ids: Optional[List[str]] = None
    ) -> None:
        """Visualize experiment tree."""
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.graph)
        
        node_colors = []
        for node in self.graph.nodes():
            if highlight_ids and node in highlight_ids:
                node_colors.append('lightred')
            else:
                exp = self.get_experiment(node)
                if exp:
                    status_colors = {
                        ExperimentStatus.PLANNED: 'lightblue',
                        ExperimentStatus.COMPLETED: 'lightgreen',
                        ExperimentStatus.ABANDONED: 'gray'
                    }
                    node_colors.append(status_colors[exp.status])
                else:
                    node_colors.append('lightgray')
        
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1000,
            arrows=True
        )
        
        if output_path:
            plt.savefig(output_path)
        plt.show()

    def _save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to file storage."""
        exp_path = self.experiments_dir / f"{experiment.id}.json"
        with open(exp_path, 'w') as f:
            json.dump(experiment.dict(), f, indent=2, default=str)

    def _update_chromadb(self, experiment: Experiment) -> None:
        """Update experiment in ChromaDB."""
        metadata = {
            "title": experiment.title,
            "type": experiment.type.value,
            "status": experiment.status.value,
            "created_at": experiment.created_at.isoformat(),
            "tags": ",".join(experiment.tags),
        }
        
        document_text = f"""
        Title: {experiment.title}
        Description: {experiment.description}
        Type: {experiment.type.value}
        Status: {experiment.status.value}
        Tags: {', '.join(experiment.tags)}
        Notes: {' | '.join(experiment.notes)}
        """
        
        self.experiment_collection.upsert(
            ids=[experiment.id],
            documents=[document_text],
            metadatas=[metadata]
        )

    def _add_to_graph(self, experiment: Experiment) -> None:
        """Add experiment to the relationship graph."""
        self.graph.add_node(experiment.id, title=experiment.title)
        if experiment.parent_id:
            self.graph.add_edge(experiment.parent_id, experiment.id)
        self._save_graph()

    def _save_graph(self) -> None:
        """Save the experiment graph."""
        data = nx.node_link_data(self.graph)
        with open(self.graph_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_graph(self) -> None:
        """Load the experiment graph."""
        if self.graph_path.exists():
            with open(self.graph_path, 'r') as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)

def main():
    """Example usage of the simplified experiment tracker."""
    # Initialize tracker
    tracker = ExperimentTracker("experiment_data")

    # Plan a root experiment
    root_exp = tracker.plan_experiment(
        title="Baseline CNN Model",
        description="Initial CNN model for image classification",
        exp_type=ExperimentType.ML,
        parameters={
            "model": "ResNet18",
            "batch_size": 32,
            "learning_rate": 0.001
        },
        target_metrics={"accuracy": 0.85},
        tags=["cnn", "baseline", "image-classification"]
    )

    # Plan a variant experiment
    variant_exp = tracker.plan_experiment(
        title="CNN with Data Augmentation",
        description="Adding data augmentation to improve accuracy",
        exp_type=ExperimentType.ML,
        parent_id=root_exp.id,
        parameters={
            "augmentation": {
                "horizontal_flip": True,
                "rotation_range": 15
            }
        },
        tags=["cnn", "data-augmentation"]
    )

    # Complete the root experiment
    tracker.complete_experiment(
        root_exp.id,
        metrics={"accuracy": 0.82, "loss": 0.35},
        conclusions="Baseline model achieved 82% accuracy. Consider data augmentation to improve performance."
    )

    # Demonstrate search
    print("\nSearching for CNN experiments:")
    cnn_experiments = tracker.search_experiments("CNN model architecture", n_results=3)
    for exp in cnn_experiments:
        print(f"- {exp.title}: {exp.description}")

    # Visualize experiments
    tracker.visualize_experiments()

if __name__ == "__main__":
    main()