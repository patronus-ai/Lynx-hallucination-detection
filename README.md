# Patronus Finetuning

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
