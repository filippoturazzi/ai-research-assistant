"""The classic AI papers (arXiv) that form the default collection."""
from collections.abc import Iterator

from rag.errors import DownloadError

PAPERS = {
    "attention_is_all_you_need": "https://arxiv.org/pdf/1706.03762",
    "retrieval_augmented_generation": "https://arxiv.org/pdf/2005.11401",
    "bert": "https://arxiv.org/pdf/1810.04805",
    "gpt3_language_models_are_few_shot_learners": "https://arxiv.org/pdf/2005.14165",
    "dense_passage_retrieval": "https://arxiv.org/pdf/2004.04906",
}


def fetch_default_papers() -> Iterator[tuple[str, bytes]]:
    import requests

    for name, url in PAPERS.items():
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DownloadError(f"Could not download '{name}': {exc}") from exc
        yield f"{name}.pdf", response.content
