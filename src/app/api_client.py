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


def _request(method: str, url: str, **kwargs) -> dict | list:
    try:
        response = requests.request(method, url, timeout=_TIMEOUT, **kwargs)
    except requests.exceptions.RequestException as exc:
        raise ApiError(
            f"Não consegui falar com a API — ela está rodando? ({exc.__class__.__name__})"
        ) from exc
    return _handle(response)


def ask(question: str, history: list[dict]) -> dict:
    return _request("POST", f"{API_URL}/ask",
                     json={"question": question, "history": history})


def upload(filename: str, data: bytes) -> dict:
    return _request("POST", f"{API_URL}/upload",
                     files={"file": (filename, data, "application/pdf")})


def send_feedback(interaction_id: int, rating: int) -> dict:
    return _request("POST", f"{API_URL}/feedback",
                     json={"interaction_id": interaction_id, "rating": rating})


def metrics() -> dict:
    return _request("GET", f"{API_URL}/metrics")


def documents() -> list:
    return _request("GET", f"{API_URL}/documents")
