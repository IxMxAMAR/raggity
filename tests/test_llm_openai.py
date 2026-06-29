import types
import pytest
import raggity.llm_openai as lo


class _Delta:
    def __init__(self, c): self.content = c
class _Choice:
    def __init__(self, c): self.delta = _Delta(c); self.message = types.SimpleNamespace(content=c)
class _Chunk:
    def __init__(self, c): self.choices = [_Choice(c)]


class _EmptyChoicesChunk:
    """Simulates a final SSE frame with choices=[] (some ollama-compat servers do this)."""
    choices = []


class _FakeStream:
    def __init__(self, parts): self._parts = parts
    def __aiter__(self):
        async def gen():
            for p in self._parts: yield _Chunk(p)
        return gen()


class _FakeStreamWithEmptyChoices:
    """Stream that yields an empty-choices chunk then a real one."""
    def __aiter__(self):
        async def gen():
            yield _EmptyChoicesChunk()
            yield _Chunk("real")
        return gen()


class _FakeCompletions:
    async def create(self, model, messages, stream=False, **kw):
        if stream: return _FakeStream(["hel", "lo"])
        return types.SimpleNamespace(choices=[_Choice("hello")])


class _FakeClient:
    def __init__(self, **kw):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions())
        self._closed = False

    async def close(self):
        self._closed = True


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


# Fix 1: aclose() closes underlying client; double-close is safe
async def test_openai_aclose_closes_client(monkeypatch):
    """aclose() must call the underlying client's close() method."""
    monkeypatch.setattr(lo, "AsyncOpenAI", _FakeClient)
    p = lo.OpenAICompatProvider(model="gpt-4o-mini")
    await p.aclose()
    assert p._client._closed is True


async def test_openai_aclose_double_close_safe(monkeypatch):
    """Calling aclose() twice must not raise."""
    monkeypatch.setattr(lo, "AsyncOpenAI", _FakeClient)
    p = lo.OpenAICompatProvider(model="gpt-4o-mini")
    await p.aclose()
    await p.aclose()  # must not raise


# Fix 2: empty choices chunk is skipped, real content still yielded
async def test_openai_stream_empty_choices_skipped(monkeypatch):
    """Stream with an empty-choices chunk must not raise IndexError; only real content yielded."""
    class _CompletionsWithEmpty:
        async def create(self, model, messages, stream=False, **kw):
            return _FakeStreamWithEmptyChoices()

    class _ClientWithEmpty:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=_CompletionsWithEmpty())
        async def close(self): pass

    monkeypatch.setattr(lo, "AsyncOpenAI", _ClientWithEmpty)
    p = lo.OpenAICompatProvider(model="gpt-4o-mini")
    out = [t async for t in p.stream("s", "p")]
    assert out == ["real"]


# Fix 3: openai backend requires key env var; ollama backend does not
def test_openai_backend_missing_key_raises(monkeypatch):
    """OpenAICompatProvider with backend=openai must raise RuntimeError when key env is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        lo.OpenAICompatProvider(model="gpt-4o-mini", require_key=True)


def test_ollama_backend_missing_key_ok(monkeypatch):
    """OpenAICompatProvider with backend=ollama (require_key=False) must not raise when key is absent."""
    monkeypatch.delenv("MY_MISSING_KEY", raising=False)
    # Should not raise even when env var is unset
    p = lo.OpenAICompatProvider(model="llama3.1", api_key_env="MY_MISSING_KEY", require_key=False)
    assert p is not None


def test_openai_from_config_missing_key_raises(monkeypatch):
    """from_config with backend=openai and missing key env must raise RuntimeError."""
    from raggity.config import GenerationConfig
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = GenerationConfig(backend="openai", model="gpt-4o-mini")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        lo.OpenAICompatProvider.from_config(cfg)


def test_ollama_from_config_missing_key_ok(monkeypatch):
    """from_config with backend=ollama and missing key env must not raise."""
    from raggity.config import GenerationConfig
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = GenerationConfig(backend="ollama", model="llama3.1")
    p = lo.OpenAICompatProvider.from_config(cfg)
    assert p is not None
