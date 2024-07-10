## Inference with mcli

We can use [Mosaic CLI](https://pypi.org/project/mosaicml-cli/) to run inference for PatronusAI/Patronus-Lynx-70B-Instruct model. To setup inference with `mcli` follow the steps below:

1. Install library package provided in `pyproject.toml` and `poetry.lock`. Follow instructions in `README.md` file.
2. Add necessary secret integrations according to documentation:
   ``` bash
   mcli init
   mcli set api-key <new-value>
   mcli create secret hf
   mcli create secret s3
   ```
3. There are two inference python files that can be used for calculating inference on HuggingFace dataset - `mcli_inference_v1.py` and `mcli_inference_v2.py`. Each of them allows inference on the datasets with the following structure:
   ``` text
   DatasetDict({
    test: Dataset({
        features: ['messages', '_id', 'LABEL'],
        num_rows: 1000
    })
   })
   
   Where row can look like:
   "LABEL": "label-1"
   "_id": "id-1"
   "messages": [{"role": "user", "content": "Who are you?"}]
   ```
4. To run inference, select one of the following `.yaml` files and fill the following parameter values:
   - files:
     - `lynx-8b-mcli-inference-v1.yaml`
     - `lynx-8b-mcli-inference-v2.yaml`
     - `lynx-70b-mcli-inference-v1.yaml`
     - `lynx-70b-mcli-inference-v2.yaml`
   - parameters:
     - `GIT_USERNAME/GIT_REPO_NAME` - points towards repository containing aforementioned inference `.py` files. Example: `patronus-ai/Lynx-hallucination-detection`.
     - `HF_USER/HF_TEST_DATASET_REPOSITORY_NAME` - HuggingFace repository storing inference input.
     - `HF_USER/HF_INFERENCE_DATASET_REPOSITORY_NAME` - HuggingFace repository storing inference results.
     - `HF_TOKEN` - Read token on HuggingFace.
     - `MCLI_RUN_NAME` - Internal name of mcli job.
     - `CLUSTER_NAME` - Cluster name provided on MCLI resources. Run `mcli get clusters` to get a list.
5. Run inference using:
    ```
    mcli run -f configs/lynx-8b-mcli-inference-v1.yaml
    ```
6. You can track your run by adding `--follow` parameter when executing `mcli run` or by using:
   ```
   mcli get runs --limit 20   
   ```
7. You can access logs by using:
   ```
   mcli logs {RUN_NAME}
   ```
   