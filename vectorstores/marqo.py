import json
import os
from typing import (
    Dict,
    List,
    Tuple
)

import marqo
import requests
from langchain.docstore.document import Document
from langchain.vectorstores.marqo import Marqo

from vectorstores.base import BaseVectorStore


class MarqoVectorStore(BaseVectorStore):
    TENSOR_FIELDS: str = ["text"]
    client: marqo.Client

    def __init__(self):
        self.client_url = os.environ["VECTOR_STORE_ENDPOINT"]
        self.collection_name = os.environ["VECTOR_COLLECTION_NAME"]
        self.embedding_model = os.environ["EMBEDDING_MODEL"]
        self.index_settings = {
            "treatUrlsAndPointersAsImages": False,
            "model": self.embedding_model,
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": self.SPLIT_LENGTH,
                "splitOverlap": self.SPLIT_OVERLAP,
                "splitMethod": "passage"
            }
        }

        if not self.client_url:
            raise ValueError("Missing environment variable VECTOR_STORE_ENDPOINT.")

        if not self.collection_name:
            raise ValueError("Missing environment variable VECTOR_COLLECTION_NAME.")

        if not self.embedding_model:
            raise ValueError("Missing environment variable EMBEDDING_MODEL.")

        self.client = marqo.Client(url=self.client_url)

    def get_client(self) -> marqo.Client:
        return self.client

    def add_documents(self, documents=List[Document], fresh_collection: bool = False) -> List[str]:

        if fresh_collection:
            try:
                self.client.index(self.collection_name).delete()
                print("Existing Index successfully deleted.")
            except:
                print("Index does not exist. Creating new index...")

            self.client.create_index(
                self.collection_name, settings_dict=self.index_settings)
            print(f"Index {self.collection_name} created.")

        docs: List[Dict[str, str]] = []
        ids = []
        for d in documents:
            doc = {
                "text": d.page_content,
                "metadata": json.dumps(d.metadata) if d.metadata else json.dumps({}),
            }
            docs.append(doc)
        chunks = list(self.chunk_list(docs, self.BATCH_SIZE))
        for chunk in chunks:
            response = self.client.index(self.collection_name).add_documents(
                documents=chunk, client_batch_size=self.BATCH_SIZE, tensor_fields=self.TENSOR_FIELDS)
            if response[0]["errors"]:
                print(f'>>>>> {response}')
                err_msg = (
                    f"Error in upload for documents in index range"
                    f"check Marqo logs."
                )
                raise RuntimeError(err_msg)
            ids += [item["_id"] for item in response[0]["items"]]

        return ids

    def similarity_search_with_score(self, query: str, collection_name: str, k: int = 20) -> List[Tuple[Document, float]]:
        docsearch = Marqo(self.client, index_name=collection_name)
        documents = docsearch.similarity_search_with_score(query, k)
        return documents

    def hybrid_search_with_score(self, query: str, collection_name: str, k: int = 20,
                                  rrf_k: int = 60) -> List[Tuple[Document, float]]:
        """Hybrid search combining tensor (dense) and lexical (BM25) via RRF.

        Uses the Marqo server REST API directly because the Python client 2.1.0
        does not expose searchMethod=HYBRID. The server (2.10+) supports it natively.

        retrievalMethod=disjunction retrieves candidates from both tensor and lexical legs
        then merges them with Reciprocal Rank Fusion (RRF). Documents without a 'keywords'
        field are handled gracefully by Marqo — they still surface via tensor or text match.

        Args:
            rrf_k: RRF constant — higher values reduce the impact of rank differences.
        """
        url = f"{self.client_url}/indexes/{collection_name}/search"
        payload = {
            "q": query,
            "limit": k,
            "searchMethod": "HYBRID",
            "hybridParameters": {
                "retrievalMethod": "disjunction",  # union of tensor + lexical candidates
                "rankingMethod": "rrf",
                "rrfK": rrf_k,
            },
        }
        response = requests.post(url, json=payload, timeout=30)
        if not response.ok:
            print(f"Hybrid search error {response.status_code}: {response.text}")
        response.raise_for_status()
        hits = response.json().get("hits", [])

        documents = []
        for hit in hits:
            text = hit.get("text", "")
            metadata_raw = hit.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            doc = Document(page_content=text, metadata=metadata)
            score = hit.get("_score", 0.0)
            documents.append((doc, score))

        return documents