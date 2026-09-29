"""
Unified LLM Client abstraction supporting OpenAI, Ollama, and Mock modes.
"""

from typing import Optional


class LLMInterface:
    def __init__(self, provider: str, model_name: str, api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
        if self.provider == "openai":
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "ollama":
            import ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_predict": max_tokens}
            )
            return response["message"]["content"].strip()

        elif self.provider == "mock":
            return f"[{self.model_name} Simulated Response] Based on context: summary generated successfully."

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")