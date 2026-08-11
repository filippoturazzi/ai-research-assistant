"""Downloads the classic AI papers (arXiv) used as the default collection."""
from pathlib import Path

import requests

from rag.ingestion.default_papers import PAPERS

DEST = Path("data/documents")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in PAPERS.items():
        path = DEST / f"{name}.pdf"
        if path.exists():
            print(f"[skip] {name} (already exists)")
            continue
        print(f"[downloading] {name} ...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    print("Done.")


if __name__ == "__main__":
    main()
