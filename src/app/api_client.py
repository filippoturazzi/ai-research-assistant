import os

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
_TIMEOUT = 120


class ApiError(Exception):
    pass


def _handle(response: requests.Response) -> dict | list:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise ApiError(detail)
    return response.json()


def ask(question: str, history: list[dict]) -> dict:
    return _handle(requests.post(f"{API_URL}/ask", timeout=_TIMEOUT,
                                 json={"question": question, "history": history}))


def upload(filename: str, data: bytes) -> dict:
    return _handle(requests.post(f"{API_URL}/upload", timeout=_TIMEOUT,
                                 files={"file": (filename, data, "application/pdf")}))


def send_feedback(interaction_id: int, rating: int) -> dict:
    return _handle(requests.post(f"{API_URL}/feedback", timeout=_TIMEOUT,
                                 json={"interaction_id": interaction_id, "rating": rating}))


def metrics() -> dict:
    return _handle(requests.get(f"{API_URL}/metrics", timeout=_TIMEOUT))


def documents() -> list:
    return _handle(requests.get(f"{API_URL}/documents", timeout=_TIMEOUT))
