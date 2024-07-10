## Inference with mcli

We can use [Mosaic CLI](https://pypi.org/project/mosaicml-cli/) to finetune any model. To setup finetuning with `mcli` follow the steps below:

1. Install library package provided in `pyproject.toml` and `poetry.lock`. Follow instructions in `README.md` file.
2. Add necessary secret integrations according to documentation:
   ``` bash
   mcli init
   mcli set api-key <new-value>
   mcli create secret hf
   mcli create secret s3
   ```
3. Finetuning uses only `.yaml` files and inputs from HuggingFace that must follow the structure:
   ``` text
   DatasetDict({
    train: Dataset({
        features: ['messages'],
        num_rows: 600
    })
    validation: Dataset({
        features: ['messages'],
        num_rows: 200
    })
   })
   
   Where row can look like:
   "messages": [{"role": "user", "content": "Who are you?"},{"role": "assistant", "content": "I am your model!"}]
   ```
4. To run finetuning, select one of the following `.yaml` files and fill the following parameter values:
   - files:
     - `lynx-8b-train.yaml`
     - `lynx-70b-train.yaml`
   - parameters:
     - `GIT_USERNAME/GIT_REPO_NAME` - points towards repository containing aforementioned inference `.py` files. Example: `patronus-ai/Lynx-hallucination-detection`.
     - `HF_USER/HF_TOKENIZER_REPOSITORY_NAME` - HuggingFace repository storing base tokenizer.
     - `HF_USER/HF_MODEL_REPOSITORY_NAME` - HuggingFace repository storing base model.
     - `CLUSTER_NAME` - Cluster name provided on MCLI resources. Run `mcli get clusters` to get a list.
     - `s3://S3_BUCKET_NAME/S3_FILEPATH` - S3 directory for output path for finetuned model.
     - `WANDB_ENTITY_NAME` - If model tracking in [Weights and Biases](https://wandb.ai/) is enabled, passes username.
     - `WANDB_PROJECT_NAME` - If model tracking in [Weights and Biases](https://wandb.ai/) is enabled, passes project name.
5. To get HuggingFace-like model files saved, uncomment 'callbacks' section of the `.yaml` file.
6. To get [Weights and Biases](https://wandb.ai/) tracking, uncomment 'loggers' and `integration_type` sections of the `.yaml` file.
7. Run inference using:
   ```
   mcli run -f configs/lynx-70b-train.yaml
   ```
8. You can track your run by adding `--follow` parameter when executing `mcli run` or by using:
   ```
   mcli get runs --limit 20   
   ```
9. You can access logs by using:
   ```
   mcli logs {RUN_NAME}
   ```
   