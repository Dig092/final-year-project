"""
Extended example demonstrating a real ML experiment workflow with the ExperimentJournal system.
This example shows training a text classification model with multiple optimization attempts.
"""
import time
from datetime import datetime, timedelta
import random

from ExperimentJournal import ExperimentJournal, ExperimentStatus, ExperimentType

def run_extended_example():
    # Initialize journal
    journal = ExperimentJournal("ml_experiment_journal")
    
    # 1. Create root experiment - Basic BERT Model
    root_exp = journal.create_experiment(
        title="Base BERT Text Classification",
        description="Initial BERT implementation for sentiment analysis",
        exp_type=ExperimentType.ML,
        parameters={
            "model": "bert-base-uncased",
            "batch_size": 32,
            "learning_rate": 2e-5,
            "epochs": 3,
            "max_length": 128
        },
        target_metrics={
            "accuracy": 0.85,
            "f1_score": 0.84,
            "training_time": 7200  # seconds
        },
        tags=["bert", "baseline", "text-classification"]
    )

    # Simulate initial training
    journal.update_experiment(
        root_exp.id,
        status=ExperimentStatus.IN_PROGRESS,
        note="Started training base BERT model"
    )
    
    # Add metrics over time to simulate training progress
    for epoch in range(3):
        acc = 0.75 + (epoch * 0.03) + random.uniform(-0.01, 0.01)
        f1 = 0.73 + (epoch * 0.03) + random.uniform(-0.01, 0.01)
        loss = 0.5 - (epoch * 0.1) + random.uniform(-0.02, 0.02)
        
        journal.add_metric(root_exp.id, "accuracy", acc, f"Epoch {epoch+1}")
        journal.add_metric(root_exp.id, "f1_score", f1, f"Epoch {epoch+1}")
        journal.add_metric(root_exp.id, "loss", loss, f"Epoch {epoch+1}")
        time.sleep(0.1)  # Simulate time passing

    journal.update_experiment(
        root_exp.id,
        status=ExperimentStatus.COMPLETED,
        note="Baseline training completed. Results below target metrics."
    )

    # 2. First optimization attempt - Learning Rate Tuning
    lr_exp = journal.create_experiment(
        title="BERT with LR Tuning",
        description="Optimizing learning rate with linear warmup",
        exp_type=ExperimentType.ML,
        parent_id=root_exp.id,
        parameters={
            "model": "bert-base-uncased",
            "batch_size": 32,
            "learning_rate": 5e-5,
            "warmup_steps": 500,
            "lr_schedule": "linear_warmup",
            "epochs": 3,
            "max_length": 128
        },
        tags=["bert", "lr-tuning", "warmup"]
    )

    journal.update_experiment(
        lr_exp.id,
        status=ExperimentStatus.IN_PROGRESS,
        note="Testing higher learning rate with warmup"
    )

    # Simulate training with better learning rate
    for epoch in range(3):
        acc = 0.78 + (epoch * 0.04) + random.uniform(-0.01, 0.01)
        f1 = 0.76 + (epoch * 0.04) + random.uniform(-0.01, 0.01)
        loss = 0.45 - (epoch * 0.12) + random.uniform(-0.02, 0.02)
        
        journal.add_metric(lr_exp.id, "accuracy", acc, f"Epoch {epoch+1}")
        journal.add_metric(lr_exp.id, "f1_score", f1, f"Epoch {epoch+1}")
        journal.add_metric(lr_exp.id, "loss", loss, f"Epoch {epoch+1}")
        time.sleep(0.1)

    journal.update_experiment(
        lr_exp.id,
        status=ExperimentStatus.COMPLETED,
        note="Improved results but still below target. Testing data augmentation."
    )

    # 3. Data Augmentation Branch
    aug_exp = journal.create_experiment(
        title="BERT with Augmentation",
        description="Adding back-translation augmentation",
        exp_type=ExperimentType.ML,
        parent_id=lr_exp.id,
        parameters={
            "model": "bert-base-uncased",
            "batch_size": 32,
            "learning_rate": 5e-5,
            "warmup_steps": 500,
            "lr_schedule": "linear_warmup",
            "epochs": 3,
            "max_length": 128,
            "augmentation": {
                "back_translation": True,
                "languages": ["fr", "de"],
                "aug_probability": 0.3
            }
        },
        tags=["bert", "augmentation", "back-translation"]
    )

    # Simulate augmented training
    journal.update_experiment(
        aug_exp.id,
        status=ExperimentStatus.IN_PROGRESS,
        note="Testing back-translation augmentation"
    )

    # Training failed due to memory issues
    journal.update_experiment(
        aug_exp.id,
        status=ExperimentStatus.FAILED,
        note="Out of memory error with augmented dataset"
    )

    # 4. Reduced Batch Size Branch
    batch_exp = journal.create_experiment(
        title="BERT with Smaller Batch",
        description="Reduced batch size to accommodate augmentation",
        exp_type=ExperimentType.ML,
        parent_id=aug_exp.id,
        parameters={
            "model": "bert-base-uncased",
            "batch_size": 16,  # Reduced
            "learning_rate": 5e-5,
            "warmup_steps": 500,
            "lr_schedule": "linear_warmup",
            "epochs": 3,
            "max_length": 128,
            "augmentation": {
                "back_translation": True,
                "languages": ["fr", "de"],
                "aug_probability": 0.3
            }
        },
        tags=["bert", "augmentation", "reduced-batch"]
    )

    journal.update_experiment(
        batch_exp.id,
        status=ExperimentStatus.IN_PROGRESS,
        note="Retrying with smaller batch size"
    )

    # Simulate successful training with augmentation
    for epoch in range(3):
        acc = 0.82 + (epoch * 0.04) + random.uniform(-0.01, 0.01)
        f1 = 0.81 + (epoch * 0.04) + random.uniform(-0.01, 0.01)
        loss = 0.4 - (epoch * 0.12) + random.uniform(-0.02, 0.02)
        
        journal.add_metric(batch_exp.id, "accuracy", acc, f"Epoch {epoch+1}")
        journal.add_metric(batch_exp.id, "f1_score", f1, f"Epoch {epoch+1}")
        journal.add_metric(batch_exp.id, "loss", loss, f"Epoch {epoch+1}")
        time.sleep(0.1)

    journal.update_experiment(
        batch_exp.id,
        status=ExperimentStatus.COMPLETED,
        note="Successfully reached target metrics!"
    )

    # 5. Parallel Experiment - DistilBERT
    distil_exp = journal.create_experiment(
        title="DistilBERT Alternative",
        description="Testing DistilBERT for faster training",
        exp_type=ExperimentType.ML,
        parent_id=root_exp.id,  # Branching from root
        parameters={
            "model": "distilbert-base-uncased",
            "batch_size": 64,  # Larger due to smaller model
            "learning_rate": 5e-5,
            "epochs": 3,
            "max_length": 128
        },
        tags=["distilbert", "optimization", "speed-test"]
    )

    journal.update_experiment(
        distil_exp.id,
        status=ExperimentStatus.IN_PROGRESS,
        note="Evaluating DistilBERT performance"
    )

    # Simulate DistilBERT training
    for epoch in range(3):
        acc = 0.76 + (epoch * 0.03) + random.uniform(-0.01, 0.01)
        f1 = 0.75 + (epoch * 0.03) + random.uniform(-0.01, 0.01)
        loss = 0.48 - (epoch * 0.1) + random.uniform(-0.02, 0.02)
        training_time = 1200 - (epoch * 100) + random.uniform(-50, 50)
        
        journal.add_metric(distil_exp.id, "accuracy", acc, f"Epoch {epoch+1}")
        journal.add_metric(distil_exp.id, "f1_score", f1, f"Epoch {epoch+1}")
        journal.add_metric(distil_exp.id, "loss", loss, f"Epoch {epoch+1}")
        journal.add_metric(distil_exp.id, "training_time", training_time, f"Epoch {epoch+1}")
        time.sleep(0.1)

    journal.update_experiment(
        distil_exp.id,
        status=ExperimentStatus.COMPLETED,
        note="Faster training but lower performance than BERT"
    )

    # 6. Final Production Model
    prod_exp = journal.create_experiment(
        title="Production BERT Model",
        description="Final model with best parameters and full training",
        exp_type=ExperimentType.ML,
        parent_id=batch_exp.id,
        parameters={
            "model": "bert-base-uncased",
            "batch_size": 16,
            "learning_rate": 5e-5,
            "warmup_steps": 500,
            "lr_schedule": "linear_warmup",
            "epochs": 5,  # Increased for final training
            "max_length": 128,
            "augmentation": {
                "back_translation": True,
                "languages": ["fr", "de"],
                "aug_probability": 0.3
            }
        },
        tags=["bert", "production", "final-model"]
    )

    # Simulate final training
    journal.update_experiment(
        prod_exp.id,
        status=ExperimentStatus.IN_PROGRESS,
        note="Training final production model"
    )

    for epoch in range(5):
        acc = 0.84 + (epoch * 0.02) + random.uniform(-0.01, 0.01)
        f1 = 0.83 + (epoch * 0.02) + random.uniform(-0.01, 0.01)
        loss = 0.35 - (epoch * 0.05) + random.uniform(-0.02, 0.02)
        
        journal.add_metric(prod_exp.id, "accuracy", acc, f"Epoch {epoch+1}")
        journal.add_metric(prod_exp.id, "f1_score", f1, f"Epoch {epoch+1}")
        journal.add_metric(prod_exp.id, "loss", loss, f"Epoch {epoch+1}")
        time.sleep(0.1)

    journal.update_experiment(
        prod_exp.id,
        status=ExperimentStatus.COMPLETED,
        note="Final model successfully trained and ready for deployment"
    )

    # Visualize the complete experiment tree
    journal.visualize_tree()

    # Print summary of all experiments
    print("\nExperiment Tree Summary:")
    print("=" * 50)
    for exp_id in journal.graph.nodes():
        exp = journal.get_experiment(exp_id)
        metrics_df = journal.get_metrics_dataframe(exp_id)
        if not metrics_df.empty:
            final_metrics = metrics_df.groupby('metric')['value'].last()
            print(f"\nExperiment: {exp.title}")
            print(f"Status: {exp.status}")
            print("Final Metrics:")
            for metric, value in final_metrics.items():
                print(f"- {metric}: {value:.4f}")
            print("-" * 30)

if __name__ == "__main__":
    run_extended_example()