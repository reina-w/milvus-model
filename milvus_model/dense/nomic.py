from typing import List
import numpy as np
from collections import defaultdict
from nomic import embed

class NomicAIEmbeddingFunction:
    def __init__(
        self,
        model_name: str = "nomic-embed-text-v1.5",
        task_type: str = "search_document",
        dimensionality: int = 768,
        **kwargs,
    ):
        self._nomic_model_meta_info = defaultdict(dict)
        self._nomic_model_meta_info[model_name]["dim"] = dimensionality  # set the dimension

        self.model_name = model_name
        self.task_type = task_type
        self.dimensionality = dimensionality
        self._encode_config = {"model": model_name, "task_type": task_type, "dimensionality": dimensionality, **kwargs}

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._encode(queries, task_type="search_query")

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._encode(documents, task_type="search_document")

    @property
    def dim(self):
        return self._nomic_model_meta_info[self.model_name]["dim"]

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode(texts, task_type=self.task_type)
    
    def _encode_query(self, query: str) -> np.array:
        return self._encode([query], task_type="search_query")[0]

    def _encode_document(self, document: str) -> np.array:
        return self._encode([document], task_type="search_document")[0]

    def _call_nomic_api(self, texts: List[str], task_type: str):
        embeddings_batch_response = embed.text(
            texts=texts,
            model=self.model_name,
            task_type=task_type,
            dimensionality=self.dimensionality,
        )
        return [np.array(embedding) for embedding in embeddings_batch_response['embeddings']]

    def _encode(self, texts: List[str], task_type: str):
        return self._call_nomic_api(texts, task_type)