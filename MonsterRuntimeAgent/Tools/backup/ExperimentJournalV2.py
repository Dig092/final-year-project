from __future__ import annotations
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
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentStatus(str, Enum):
    """Status states for experiments."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"

class ExperimentType(str, Enum):
    """Types of experiments that can be tracked."""
    ML = "machine_learning"
    DATA_ANALYSIS = "data_analysis"
    AB_TEST = "ab_testing"
    OTHER = "other"

class MetricEntry(BaseModel):
    """Track a single metric value with timestamp."""
    value: float
    timestamp: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = None

class ExperimentResult(BaseModel):
    """Store experiment results and metrics."""
    metrics: Dict[str, List[MetricEntry]] = Field(default_factory=dict)
    artifacts_path: Optional[str] = None
    conclusions: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class Experiment(BaseModel):
    """Core experiment tracking model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    title: str
    description: str
    type: ExperimentType
    status: ExperimentStatus = ExperimentStatus.PLANNED
    
    # Configuration and parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Progress tracking
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results and metrics
    target_metrics: Dict[str, float] = Field(default_factory=dict)
    results: List[ExperimentResult] = Field(default_factory=list)
    
    # Notes and documentation
    notes: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True

class ExperimentTracker:
    """Manage and track experiments with their relationships and progress."""
    
    def __init__(self, storage_dir: Union[str, Path]):
        """Initialize the experiment tracker with file and ChromaDB storage."""
        self.storage_dir = Path(storage_dir)
        self.experiments_dir = self.storage_dir / "experiments"
        self.artifacts_dir = self.storage_dir / "artifacts"
        self.graph_path = self.storage_dir / "experiment_graph.json"
        self.chromadb_path = self.storage_dir / "chromadb"
        
        # Create necessary directories
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.chromadb_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment graph
        self.graph = nx.DiGraph()
        self._load_graph()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chromadb_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get the collection for experiments
        self.experiment_collection = self.chroma_client.get_or_create_collection(
            name="experiments",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        
        logger.info(f"Initialized ExperimentTracker with storage at {storage_dir}")

    def create_experiment(
        self,
        title: str,
        description: str,
        exp_type: ExperimentType,
        parent_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        target_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None
    ) -> Experiment:
        """Create a new experiment and store it."""
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
        
        logger.info(f"Created new experiment: {experiment.id} - {experiment.title}")
        return experiment

    def update_experiment(
        self,
        exp_id: str,
        status: Optional[ExperimentStatus] = None,
        parameters: Optional[Dict[str, Any]] = None,
        note: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> Experiment:
        """Update an existing experiment's status, parameters, notes, or metrics."""
        experiment = self.get_experiment(exp_id)
        if not experiment:
            raise ValueError(f"Experiment {exp_id} not found")

        if status:
            experiment.status = status
            if status == ExperimentStatus.IN_PROGRESS:
                experiment.started_at = datetime.now()
            elif status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
                experiment.completed_at = datetime.now()

        if parameters:
            experiment.parameters.update(parameters)

        if note:
            experiment.notes.append(f"{datetime.now()}: {note}")

        if metrics:
            result = ExperimentResult()
            for metric_name, value in metrics.items():
                result.metrics[metric_name] = [MetricEntry(value=value)]
            experiment.results.append(result)

        self._save_experiment(experiment)
        self._update_chromadb(experiment)
        
        logger.info(f"Updated experiment: {experiment.id}")
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

    def get_similar_experiments(
        self,
        exp_id: str,
        n_results: int = 5
    ) -> List[Experiment]:
        """Find experiments similar to the given one."""
        experiment = self.get_experiment(exp_id)
        if not experiment:
            raise ValueError(f"Experiment {exp_id} not found")
        
        query_text = f"{experiment.title} {experiment.description}"
        results = self.experiment_collection.query(
            query_texts=[query_text],
            n_results=n_results + 1
        )
        
        similar_experiments = []
        if results and results['ids']:
            for result_id in results['ids'][0]:
                if result_id != exp_id:
                    similar_exp = self.get_experiment(result_id)
                    if similar_exp:
                        similar_experiments.append(similar_exp)
        
        return similar_experiments[:n_results]

    def get_experiment_tree(self, root_id: Optional[str] = None) -> nx.DiGraph:
        """Get the experiment dependency tree."""
        if root_id:
            return nx.dfs_tree(self.graph, root_id)
        return self.graph

    def visualize_tree(
        self,
        output_path: Optional[Path] = None,
        highlight_ids: Optional[List[str]] = None
    ) -> None:
        """Visualize the experiment tree."""
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes with colors based on status
        node_colors = []
        for node in self.graph.nodes():
            if highlight_ids and node in highlight_ids:
                node_colors.append('lightred')
            else:
                exp = self.get_experiment(node)
                if exp:
                    status_colors = {
                        ExperimentStatus.PLANNED: 'lightblue',
                        ExperimentStatus.IN_PROGRESS: 'yellow',
                        ExperimentStatus.COMPLETED: 'lightgreen',
                        ExperimentStatus.FAILED: 'red',
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

    def get_metrics_dataframe(self, exp_id: str) -> pd.DataFrame:
        """Get experiment metrics as a pandas DataFrame."""
        experiment = self.get_experiment(exp_id)
        if not experiment:
            raise ValueError(f"Experiment {exp_id} not found")

        metrics_data = []
        for result in experiment.results:
            for metric_name, entries in result.metrics.items():
                for entry in entries:
                    metrics_data.append({
                        'metric': metric_name,
                        'value': entry.value,
                        'timestamp': entry.timestamp,
                        'notes': entry.notes
                    })
        
        return pd.DataFrame(metrics_data)

    def _save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to file storage."""
        exp_path = self.experiments_dir / f"{experiment.id}.json"
        with open(exp_path, 'w') as f:
            json.dump(experiment.dict(), f, indent=2, default=str)

    def _update_chromadb(self, experiment: Experiment) -> None:
        """Update or insert experiment in ChromaDB."""
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
        """Save the experiment graph to storage."""
        data = nx.node_link_data(self.graph)
        with open(self.graph_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_graph(self) -> None:
        """Load the experiment graph from storage."""
        if self.graph_path.exists():
            with open(self.graph_path, 'r') as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)

def main():
    """Example usage of the experiment tracker."""
    # Initialize tracker
    tracker = ExperimentTracker("experiment_data")

    # Create a root experiment
    root_exp = tracker.create_experiment(
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

    # Update experiment with some results
    tracker.update_experiment(
        root_exp.id,
        status=ExperimentStatus.IN_PROGRESS,
        note="Started training with default parameters"
    )
    
    # Add some metric results
    tracker.update_experiment(
        root_exp.id,
        metrics={"accuracy": 0.82, "loss": 0.35}
    )

    # Create a variant experiment
    variant_exp = tracker.create_experiment(
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

    # Demonstrate search functionality
    print("\nSearching for CNN experiments:")
    cnn_experiments = tracker.search_experiments("CNN model architecture", n_results=3)
    for exp in cnn_experiments:
        print(f"- {exp.title}: {exp.description}")

    # Find similar experiments
    print("\nFinding similar experiments to the root experiment:")
    similar_experiments = tracker.get_similar_experiments(root_exp.id, n_results=2)
    for exp in similar_experiments:
        print(f"- {exp.title}: {exp.description}")

    # Show metrics
    metrics_df = tracker.get_metrics_dataframe(root_exp.id)
    print("\nMetrics Summary:")
    print(metrics_df)

    # Visualize the experiment tree
    tracker.visualize_tree()

if __name__ == "__main__":
    main()