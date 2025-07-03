from monsterapi.client import DreamboothDeploymentConfig, \
                            CustomImageParams, \
                            LLMServingParams, \
                            DreamboothServiceParamsFinetuning, \
                            WhisperServiceParamsFinetuning, \
                            LLMServiceParamsFinetuning
                            
from typing import Optional, Dict, Any
import requests
import logging
import time
import os

logger = logging.getLogger(__name__)

# Monster API Client with Pydantic models for inputs
class MonsterDeployClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = 'https://api.monsterapi.ai/v1'):
        if api_key is None:
            api_key = os.getenv("MONSTER_API_KEY")
            if not api_key:
                raise ValueError("API Key must be provided")
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

    def _post(self, url: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(url, headers=self.headers, json=json_data)
        response.raise_for_status()
        return response.json()

    # Individual deployment methods using Pydantic models
    def deploy_llm(self, inputModel: LLMServingParams) -> Dict[str, Any]:
        """Deploy an LLM using Monster API."""
        url = f"{self.base_url}/deploy/llm"
        return self._post(url, inputModel.dict())

    def deploy_custom_image(self, inputModel: CustomImageParams) -> Dict[str, Any]:
        """Deploy a custom Docker image."""
        url = f"{self.base_url}/deploy/custom_image"
        return self._post(url, inputModel.dict())

    def deploy_dreambooth(self, inputModel: DreamboothDeploymentConfig) -> Dict[str, Any]:
        """Deploy a Dreambooth instance."""
        url = f"{self.base_url}/deploy/sdxl-dreambooth"
        return self._post(url, inputModel.dict())

    # Fine-tuning methods
    def finetune_whisper(self, inputModel: WhisperServiceParamsFinetuning) -> Dict[str, Any]:
        """Fine-tune a Whisper model."""
        url = f"{self.base_url}/finetune/speech2text/whisper"
        return self._post(url, inputModel.dict())

    def finetuning_llm(self, inputModel: LLMServiceParamsFinetuning) -> Dict[str, Any]:
        """ Finetune a LLM model """
        url = f"{self.base_url}/finetune/llm"
        return self._post(url, inputModel.dict())

    def finetuning_sdxl_dreambooth(self, inputModel: DreamboothServiceParamsFinetuning) -> Dict[str, Any]:
        """ Fintune a sd model dreambooth mode """
        url = f"{self.base_url}/finetune/text2image/sdxl-dreambooth"
        return self._post(url, inputModel.dict())

    # Example helper methods for handling responses, retries, etc.
    def wait_for_completion(self, process_id: str, timeout: int = 100) -> Dict[str, Any]:
        """Wait for a deployment process to complete."""
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Process {process_id} timed out after {timeout} seconds.")
            status = self.get_status(process_id)
            if status['status'].lower() == 'completed':
                return status['result']
            elif status['status'].lower() == 'failed':
                raise RuntimeError(f"Process {process_id} failed: {status}")
            time.sleep(10)

    def get_status(self, process_id: str) -> Dict[str, Any]:
        """Check the status of a process."""
        url = f"{self.base_url}/status/{process_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def terminate_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Terminate a running deployment."""
        url = f"{self.base_url}/deploy/terminate"
        payload = {"deployment_id": deployment_id, "actor": "user"}
        return self._post(url, payload)

                            
