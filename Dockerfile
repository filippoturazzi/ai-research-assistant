FROM python:3.12-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app
RUN chown user /app
USER user

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
COPY --chown=user scripts ./scripts

# CPU-only torch (the default PyPI wheel bundles unused CUDA libraries)
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --user .

# Bake the default collection and index into the image
# (downloads the papers and the embedding model at build time)
RUN python scripts/download_papers.py && python scripts/build_index.py

# Pre-download the cross-encoder so startup needs no network
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user start.sh ./
EXPOSE 7860
CMD ["bash", "start.sh"]
