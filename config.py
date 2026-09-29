"""
Configuration file for model definitions, default parameters, and execution settings.
"""

DEFAULT_TOP_K = 5
DEFAULT_RRF_K = 60
DEFAULT_DATASET_LIMIT = 5000

LLM_CONFIGURATIONS = {
    "GPT-4": {
        "provider": "openai",
        "model_name": "gpt-4"
    },
    "LLaMA-2": {
        "provider": "ollama",
        "model_name": "llama2"
    },
    "Mistral-7B": {
        "provider": "ollama",
        "model_name": "mistral"
    }
}

LLM_MOCK_CONFIGURATIONS = {
    "GPT-4": {
        "provider": "mock",
        "model_name": "gpt-4"
    },
    "LLaMA-2": {
        "provider": "mock",
        "model_name": "llama2"
    },
    "Mistral-7B": {
        "provider": "mock",
        "model_name": "mistral"
    }
}