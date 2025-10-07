from typing import Dict
import json

_ats = None

def init_ats(ats_instance):
    """Initialize the global ATS instance"""
    global _ats
    _ats = ats_instance

def update_stage_status(stage_number: int, stage_name: str = None, 
                stage_accomplished: str = "No", stage_summary: str = "Not started") -> str:
    """Update a stage in the ATS system
    Args:
        stage_number (int): The stage number (1-3)
        stage_name (str, optional): Name of the stage
        stage_accomplished (str, optional): Yes or No
        stage_summary (str, optional): Brief summary of the stage status
    Returns:
        str: JSON string of updated stages
    """
    if not _ats:
        raise RuntimeError("ATS not initialized. Call init_ats first.")
    
    stage_info = {
        "stage_number": int(stage_number),
        "stage_name": stage_name or f"Stage {stage_number}",
        "stage_accomplished": stage_accomplished,
        "stage_summary": stage_summary
    }
    result = _ats.add_or_update_stage(stage_info)
    return json.dumps(result, indent=2)

def add_subtask(stage_number: int, subtask_name: str, 
                accomplished: str = "No", summary: str = "Not started") -> str:
    """Add a subtask to a specific stage
    Args:
        stage_number (int): The stage number (1-3)
        subtask_name (str): Name of the subtask
        accomplished (str, optional): Yes or No
        summary (str, optional): Brief summary of the subtask status
    Returns:
        str: JSON string of updated stages
    """
    if not _ats:
        raise RuntimeError("ATS not initialized. Call init_ats first.")
    
    subtask = {
        "name": subtask_name,
        "accomplished": accomplished,
        "summary": summary
    }
    result = _ats.add_subtask(int(stage_number), subtask)
    return json.dumps(result, indent=2)

def update_subtask(stage_number: int, subtask_index: int, 
                   accomplished: str = None, summary: str = None) -> str:
    """Update a specific subtask's status
    Args:
        stage_number (int): The stage number (1-3)
        subtask_index (int): Index of the subtask to update
        accomplished (str, optional): Yes or No
        summary (str, optional): Brief summary of the subtask status
    Returns:
        str: JSON string of updated stages
    """
    if not _ats:
        raise RuntimeError("ATS not initialized. Call init_ats first.")
    
    status = {}
    if accomplished is not None:
        status["accomplished"] = accomplished
    if summary is not None:
        status["summary"] = summary
    result = _ats.update_subtask(int(stage_number), subtask_index, status)
    return json.dumps(result, indent=2)

def get_stage_status() -> str:
    """Get the current status of all stages and their subtasks
    Returns:
        str: JSON string of all stages and subtasks
    """
    if not _ats:
        raise RuntimeError("ATS not initialized. Call init_ats first.")
    
    result = _ats.get_current_status()
    return json.dumps(result, indent=2)