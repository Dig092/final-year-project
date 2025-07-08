

## Installation

### Dependencies

python3.12, docker

### Steps

1. ```bash MonsterRuntimeAgent/install_deps.sh```

2. ```python3 -m pip install -r MonsterRuntimeAgent/requirements.txt```

3. ```cp OAI_CONFIG_LIST_EXAMPLE OAI_CONFIG_LIST```

4. Update OAI_CONFIG_LIST to use actual API Keys Only replcae openaAI API key dont use claude unless in prod.

5. ```export OPENAI_API_KEY="<Replace with openAI API Key>"```

6. ```export GOOGLE_SEARCH_API_KEY="Replace with GOOGLE SEARCH API KEY ask vikas/gaurav"```

7. ```export GOOGLE_SEARCH_CSE_ID="Replcae with CSE ID ask vikas/gaurav"```

8.  ```bash
    cd worker_app/
    bash build_and_run.sh cpu
    ```

9.  ```bash
    cd monster_dockerapi/
    bash build_and_run.sh
    ```

10. Verify docker mgmt api docker running fine on local

11. Configure AWS Credentials

12. ```python3 run.py ``` # Interact with agent
