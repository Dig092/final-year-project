from typing import Dict, List
import json
import os

class ATSTool:
    def __init__(self, thread_id: str):
        """
        Initialize Automated tracking system (ATS) with file storage
        """
        self.stages = []
        self.filename = f"/tmp/{thread_id}.json"
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create the JSON file if it doesn't exist"""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as f:
                json.dump([], f)

    def _read_stages(self) -> List[Dict]:
        """Read stages from file"""
        with open(self.filename, 'r') as f:
            return json.load(f)

    def _write_stages(self, stages: List[Dict]):
        """Write stages to file"""
        with open(self.filename, 'w') as f:
            json.dump(stages, f, indent=2)

    def get_stage_by_number(self, stage_number: int) -> Dict:
        """Get stage by number from file"""
        stages = self._read_stages()
        for stage in stages:
            if int(stage.get("stage_number", 0)) == stage_number:
                return stage
        return None

    def add_or_update_stage(self, stage_info: Dict) -> List[Dict]:
        """Add or update a stage in the file"""
        stages = self._read_stages()
        stage_number = int(stage_info.get("stage_number"))
        stage = {
            "stage_number": stage_number,
            "stage_name": stage_info.get("stage_name", f"Stage {stage_number}"),
            "stage_accomplished": stage_info.get("stage_accomplished", "No"),
            "stage_summary": stage_info.get("stage_summary", "Not started"),
            "subtasks": stage_info.get("subtasks", [])
        }
        
        updated = False
        for i, existing_stage in enumerate(stages):
            if int(existing_stage.get("stage_number", 0)) == stage_number:
                if not stage["subtasks"] and "subtasks" in existing_stage:
                    stage["subtasks"] = existing_stage["subtasks"]
                stages[i] = stage
                updated = True
                break
        
        if not updated:
            stages.append(stage)
        
        self._write_stages(stages)
        return stages

    def add_subtask(self, stage_number: int, subtask: Dict) -> List[Dict]:
        """Add a subtask to a specific stage"""
        stages = self._read_stages()
        stage = self.get_stage_by_number(stage_number)
        
        if not stage:
            stage = {
                "stage_number": stage_number,
                "stage_name": f"Stage {stage_number}",
                "stage_accomplished": "No",
                "stage_summary": "Stage initialized",
                "subtasks": []
            }
            stages.append(stage)
        
        for stage in stages:
            if int(stage.get("stage_number", 0)) == stage_number:
                if "subtasks" not in stage:
                    stage["subtasks"] = []
                subtask_with_status = {
                    "subtask_name": subtask.get("name"),
                    "accomplished": subtask.get("accomplished", "No"),
                    "summary": subtask.get("summary", "Not started")
                }
                stage["subtasks"].append(subtask_with_status)
                break
        
        self._write_stages(stages)
        return stages

    def update_subtask(self, stage_number: int, subtask_index: int, status: Dict) -> List[Dict]:
        """Update a specific subtask's status"""
        stages = self._read_stages()
        for stage in stages:
            if int(stage.get("stage_number", 0)) == stage_number:
                if "subtasks" in stage and 0 <= subtask_index < len(stage["subtasks"]):
                    stage["subtasks"][subtask_index].update(status)
                break
        
        self._write_stages(stages)
        return stages

    def get_current_status(self) -> List[Dict]:
        """Get current status of all stages"""
        stages = self._read_stages()
        return sorted(stages, key=lambda x: int(x.get("stage_number", 0)))