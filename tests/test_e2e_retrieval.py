import pytest

from rag.config import EMBEDDING_DIM
from rag.models import Chunk
from rag.retrieval.embedder import Embedder
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import HybridRetriever
from rag.retrieval.store import IndexStore


@pytest.mark.integration
def test_semantic_retrieval_end_to_end():
    docs = {
        "transformers": "The Transformer architecture relies entirely on self-attention "
                        "mechanisms to model relationships between tokens in a sequence.",
        "cnn": "Convolutional neural networks apply learned filters over images to "
               "detect visual patterns such as edges and textures.",
        "rl": "Reinforcement learning agents interact with an environment and learn "
              "policies that maximize cumulative reward.",
    }
    chunks = [Chunk(chunk_id=f"{k}:0", doc_id=k, doc_title=k.title(), page=1,
                    position=0, text=v) for k, v in docs.items()]
    embedder = Embedder()
    store = IndexStore(dim=EMBEDDING_DIM)
    store.add(chunks, embedder.embed_texts([c.text for c in chunks]))

    retriever = HybridRetriever(store, embedder, Reranker(), top_k=2)
    results = retriever.retrieve("Which architecture is based on self-attention?")
    assert results[0].chunk.doc_id == "transformers"
