# Lynx Hallucination Detection

### Setup
Install local environment using pyenv and `.python-version`, `poetry.lock`, `pyproject.toml` files.
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip 
pip install poetry==1.8.1
poetry update
poetry install 
```

Install pre-commit in your teminal and run 
```bash
pre-commit install
```

Copy `.env.example` to `.env` and replace values for environmental variables.

### Contents
- Inference and Finetuning setups on mcli and vLLM. Details:
  - mcli/mcli_finetuning.md
  - mcli/mcli_inference.md
  - mcli/vllm_inference.md


### Communication

* If you found a bug, open an issue.


### Authors
* [Rebecca Qian](https://www.linkedin.com/in/rebeccaqian/) - email: [rebecca@patronus.ai](mailto:rebecca@patronus.ai)
* [Sunitha Ravi](https://www.linkedin.com/in/selvansunitha/) - email: [sunitha@patronus.ai](mailto:sunitha@patronus.ai)
* [Bartosz Mielczarek](https://www.linkedin.com/in/bartosz-mielczarek-647346117/) - email: [bartek@patronus.ai](mailto:bartek@patronus.ai)


Want to talk about LLM use cases, visit our webpage [PatronusAI](https://www.patronus.ai/).
