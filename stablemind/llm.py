class LLMClient:
    """Replace this with OpenAI/Claude/local model later."""

    def __init__(self):
        pass

    @staticmethod
    def dummy():
        return _DummyLLM()

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class _DummyLLM(LLMClient):
    def generate(self, prompt: str) -> str:
        # placeholder so your pipeline runs end-to-end
        return "🌙✨ — R (dummy reply; plug in a real LLMClient)"