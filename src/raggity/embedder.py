from __future__ import annotations

from abc import ABC, abstractmethod

from .registry import register

_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "rocm": ["ROCMExecutionProvider", "CPUExecutionProvider"],
}


class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 provider: str = "cpu",
                 batch_size: int = 256,
                 parallel: int = 0) -> None:
        from fastembed import TextEmbedding

        providers = _PROVIDERS.get(provider, _PROVIDERS["cpu"])
        self._model = TextEmbedding(model_name=model_name, providers=providers)
        self._batch_size = batch_size
        self._parallel = parallel
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(
            texts, batch_size=self._batch_size, parallel=self._parallel)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()


register("embedder", "fastembed", "raggity.embedder:FastEmbedEmbedder")
