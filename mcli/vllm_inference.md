## Inference with vLLM

We can use [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) to run inference for PatronusAI/Patronus-Lynx-70B-Instruct model. To setup inference with vLLM follow the steps below:

1. Download model weights onto a local directory.
2. Install vLLM (if not installed using the poetry package):
    ```
    pip install vllm==0.4.3
    ```

3. Set HOST_IP env variable, if you are using docker:
    ```
    export HOST_IP=$(hostname -i)
    ```

4. To run offline batch inference, create a jsonl file in the following format:

    ``` 
    {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

    {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
    ```
5. Run inference using:
    ```
    python -m vllm.entrypoints.openai run_batch -i drop_batch.jsonl -o drop_results.jsonl --model llama-3-70b --tensor-parallel-size 8
    ```

If you want to run online inference using the model, you can bring up the model server using:

```
python -m vllm.entrypoints.openai.api_server \
--model llama-3-70b --tensor-parallel-size 8
```

and query the endpoint:

```
curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
     "model": "llama-3-70b",
     "messages": [
         {"role": "user", "content": "You are a helpful assistant.Who won the world series in 2020?"},
     ]
}'
```
