import types
import raggity.llm_openai as lo


class _Delta:
    def __init__(self, c): self.content = c
class _Choice:
    def __init__(self, c): self.delta = _Delta(c); self.message = types.SimpleNamespace(content=c)
class _Chunk:
    def __init__(self, c): self.choices = [_Choice(c)]


class _FakeStream:
    def __init__(self, parts): self._parts = parts
    def __aiter__(self):
        async def gen():
            for p in self._parts: yield _Chunk(p)
        return gen()


class _FakeCompletions:
    async def create(self, model, messages, stream=False, **kw):
        if stream: return _FakeStream(["hel", "lo"])
        return types.SimpleNamespace(choices=[_Choice("hello")])
class _FakeClient:
    def __init__(self, **kw): self.chat = types.SimpleNamespace(completions=_FakeCompletions())


async def test_openai_complete(monkeypatch):
    monkeypatch.setattr(lo, "AsyncOpenAI", _FakeClient)
    p = lo.OpenAICompatProvider(model="gpt-4o-mini")
    assert (await p.complete("s", "p")).strip() == "hello"

async def test_openai_stream(monkeypatch):
    monkeypatch.setattr(lo, "AsyncOpenAI", _FakeClient)
    p = lo.OpenAICompatProvider(model="gpt-4o-mini")
    assert [t async for t in p.stream("s", "p")] == ["hel", "lo"]

def test_ollama_from_config_defaults_localhost():
    from raggity.config import GenerationConfig
    monkey = GenerationConfig(backend="ollama", model="llama3.1")
    p = lo.OpenAICompatProvider.from_config(monkey)
    assert "11434" in p.base_url
