"""Baixa os papers clássicos de IA (arXiv) usados como coleção padrão."""
from pathlib import Path

import requests

PAPERS = {
    "attention_is_all_you_need": "https://arxiv.org/pdf/1706.03762",
    "retrieval_augmented_generation": "https://arxiv.org/pdf/2005.11401",
    "bert": "https://arxiv.org/pdf/1810.04805",
    "gpt3_language_models_are_few_shot_learners": "https://arxiv.org/pdf/2005.14165",
    "dense_passage_retrieval": "https://arxiv.org/pdf/2004.04906",
}

DEST = Path("data/documents")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in PAPERS.items():
        path = DEST / f"{name}.pdf"
        if path.exists():
            print(f"[skip] {name} (já existe)")
            continue
        print(f"[baixando] {name} ...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    print("Concluído.")


if __name__ == "__main__":
    main()
