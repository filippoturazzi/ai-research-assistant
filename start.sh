#!/usr/bin/env bash
set -e

uvicorn "api.main:create_app" --factory --host 127.0.0.1 --port 8000 &

python - <<'EOF'
import time
import urllib.request

for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        print("API is up.")
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("API did not start within 60s")
EOF

exec streamlit run src/app/Home.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
