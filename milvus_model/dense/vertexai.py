from typing import List, Optional
import numpy as np
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from collections import defaultdict

class VertexAIEmbeddingFunction:
    def __init__(
        self,
        model_name: str = "text-embedding-004",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = 256,
        task: str = "SEMANTIC_SIMILARITY",
        **kwargs,
    ):
        self._vertexai_model_meta_info = defaultdict(dict)
        self._vertexai_model_meta_info["text-embedding-004"]["dim"] = 256

        self._model_config = dict({"api_key": api_key, "base_url": base_url}, **kwargs)
        additional_encode_config = {}
        if dimensions is not None:
            additional_encode_config = {"dimensions": dimensions}
            self._vertexai_model_meta_info[model_name]["dim"] = dimensions

        self._encode_config = {"model": model_name, "task": task, **additional_encode_config}
        self.model_name = model_name
        self.client = TextEmbeddingModel.from_pretrained(model_name)

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._encode(queries, task="RETRIEVAL_QUERY")

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._encode(documents, task="RETRIEVAL_DOCUMENT")

    @property
    def dim(self):
        return self._vertexai_model_meta_info[self.model_name]["dim"]

    def __call__(self, texts: List[str], task: str = "SEMANTIC_SIMILARITY") -> List[np.array]:
        return self._encode(texts, task)

    def _encode_query(self, query: str) -> np.array:
        return self._encode([query], task="RETRIEVAL_QUERY")[0]

    def _encode_document(self, document: str) -> np.array:
        return self._encode([document], task="RETRIEVAL_DOCUMENT")[0]

    def _call_vertexai_api(self, texts: List[str]):
        inputs = [TextEmbeddingInput(text, self._encode_config["task"]) for text in texts]
        embeddings = self.client.get_embeddings(inputs, **self._encode_config)
        return [np.array(embedding.values) for embedding in embeddings]

    def _encode(self, texts: List[str]):
        return self._call_vertexai_api(texts)