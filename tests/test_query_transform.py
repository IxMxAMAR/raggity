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


async def test_generate_query_variations_exact_count(monkeypatch):
    async def _fake_query(prompt, options):
        yield _AssistantMessage("v1\nv2\nv3\nv4\nv5")
    monkeypatch.setattr(qt, "query", _fake_query)
    monkeypatch.setattr(qt, "AssistantMessage", _AssistantMessage)
    out = await qt.generate_query_variations("orig", n=2)
    assert out == ["orig", "v1", "v2"]


async def test_hyde_returns_passage(monkeypatch):
    async def _fq(prompt, options):
        yield _AssistantMessage("Backups are written nightly to the NAS device.")
    monkeypatch.setattr(qt, "query", _fq)
    monkeypatch.setattr(qt, "AssistantMessage", _AssistantMessage)
    out = await qt.generate_hyde_document("how are backups done?")
    assert "NAS" in out


async def test_step_back_returns_question(monkeypatch):
    async def _fq(prompt, options):
        yield _AssistantMessage("What is the overall backup strategy?")
    monkeypatch.setattr(qt, "query", _fq)
    monkeypatch.setattr(qt, "AssistantMessage", _AssistantMessage)
    out = await qt.generate_step_back_question("when did the last NAS backup run?")
    assert out.strip().endswith("?")


async def test_decompose_returns_subquestions(monkeypatch):
    async def _fq(prompt, options):
        yield _AssistantMessage("What backup software is used?\nHow often do backups run?\nWhere are backups stored?")
    monkeypatch.setattr(qt, "query", _fq)
    monkeypatch.setattr(qt, "AssistantMessage", _AssistantMessage)
    out = await qt.decompose_question("explain the backup setup", n=3)
    assert len(out) == 3 and "How often do backups run?" in out
