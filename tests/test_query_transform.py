import raggity.query_transform as qt


class _Block:
    def __init__(self, text): self.text = text

class _AssistantMessage:
    def __init__(self, text): self.content = [_Block(text)]


async def test_generate_query_variations_parses_lines(monkeypatch):
    async def _fake_query(prompt, options):
        yield _AssistantMessage("how do backups work\nbackup process details\nNAS backup schedule")
    monkeypatch.setattr(qt, "query", _fake_query)
    monkeypatch.setattr(qt, "AssistantMessage", _AssistantMessage)
    out = await qt.generate_query_variations("how are backups done?", n=3)
    assert out[0] == "how are backups done?"      # original first
    assert "backup process details" in out
    assert len(out) <= 4                           # original + up to 3


async def test_generate_query_variations_deduplicates_original(monkeypatch):
    async def _fake_query(prompt, options):
        yield _AssistantMessage("how are backups done?\nalternative phrasing\nthird option")
    monkeypatch.setattr(qt, "query", _fake_query)
    monkeypatch.setattr(qt, "AssistantMessage", _AssistantMessage)
    out = await qt.generate_query_variations("how are backups done?", n=3)
    assert out.count("how are backups done?") == 1  # not duplicated


async def test_generate_query_variations_respects_n(monkeypatch):
    async def _fake_query(prompt, options):
        yield _AssistantMessage("line1\nline2\nline3\nline4\nline5")
    monkeypatch.setattr(qt, "query", _fake_query)
    monkeypatch.setattr(qt, "AssistantMessage", _AssistantMessage)
    out = await qt.generate_query_variations("q", n=2)
    # original + up to 2 variations = at most 3
    assert len(out) <= 3
