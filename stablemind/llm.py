# stablemind/llm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()  # loads OPENAI_API_KEY from .env


@dataclass
class LLMConfig:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.7
    max_output_tokens: int = 400


class LLMClient:
    """Base class: StableMind calls .generate(prompt)"""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    @staticmethod
    def dummy() -> "LLMClient":
        return _DummyLLM()

    @staticmethod
    def openai(config: Optional[LLMConfig] = None) -> "LLMClient":
        return OpenAILLMClient(config=config or LLMConfig())


class _DummyLLM(LLMClient):
    def generate(self, prompt: str) -> str:
        return "🌙✨ — R (dummy reply; plug in a real LLMClient)"


class OpenAILLMClient(LLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI()  # reads OPENAI_API_KEY

    def generate(self, prompt: str) -> str:
        # Responses API (current)
        resp = self.client.responses.create(
            model=self.config.model,
            input=prompt,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        text = getattr(resp, "output_text", None)
        if not text:
            # Fail loudly: if OpenAI changes payload or something is wrong, you should know.
            raise RuntimeError(f"OpenAI response had no output_text. Raw: {resp}")
        return text.strip()