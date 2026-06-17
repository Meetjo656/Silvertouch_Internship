import os
import time
from typing import Optional
from openai import OpenAI, APIError, APITimeoutError

class QwenGenerator:
    """
    OpenAI-compatible client wrapper to generate SQL via a vLLM endpoint.
    Handles configuration, retries, and generation settings.
    """
    def __init__(self, 
                 base_url: str = "http://localhost:8000/v1", 
                 api_key: str = "EMPTY", 
                 model_name: str = "Qwen2.5-Coder-7B-Instruct",
                 temperature: float = 0.0,
                 top_p: float = 0.9,
                 max_tokens: int = 512,
                 max_retries: int = 3,
                 timeout: float = 60.0):
        # We allow overriding via env vars
        base_url = os.environ.get("VLLM_API_BASE", base_url)
        api_key = os.environ.get("VLLM_API_KEY", api_key)
        self.model_name = os.environ.get("VLLM_MODEL_NAME", model_name)
        
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate(self, prompt: str) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    stop=["---"] # Stop generating if it starts a new section
                )
                if response.choices:
                    generated_text = response.choices[0].text.strip()
                    # Clean up any potential markdown backticks that models tend to output
                    if generated_text.startswith("```sql"):
                        generated_text = generated_text[6:]
                    if generated_text.startswith("```"):
                        generated_text = generated_text[3:]
                    if generated_text.endswith("```"):
                        generated_text = generated_text[:-3]
                    return generated_text.strip()
            except (APIError, APITimeoutError) as e:
                print(f"Qwen API error on attempt {attempt + 1}/{self.max_retries}: {e}")
                time.sleep(2 ** attempt)  # exponential backoff
            except Exception as e:
                print(f"Unknown error on attempt {attempt + 1}/{self.max_retries}: {e}")
                time.sleep(2 ** attempt)
        
        return None
