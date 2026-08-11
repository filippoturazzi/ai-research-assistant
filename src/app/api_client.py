import os

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
_TIMEOUT = 120


class ApiError(Exception):
    pass


class ApiConnectionError(ApiError):
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
    kwargs.setdefault("timeout", _TIMEOUT)
    try:
        response = requests.request(method, url, **kwargs)
    except requests.exceptions.RequestException as exc:
        raise ApiConnectionError(
            f"Could not reach the API — is it running? ({exc.__class__.__name__})"
        ) from exc
    return _handle(response)


def ask(question: str, history: list[dict], language: str = "en") -> dict:
    return _request("POST", f"{API_URL}/ask",
                     json={"question": question, "history": history,
                           "language": language})


def upload(files: list[tuple[str, bytes]]) -> dict:
    parts = [("files", (name, data, "application/pdf")) for name, data in files]
    return _request("POST", f"{API_URL}/upload", files=parts)


def remove_document(doc_id: str) -> dict:
    return _request("DELETE", f"{API_URL}/documents/{doc_id}")


def reset_documents() -> dict:
    return _request("POST", f"{API_URL}/documents/reset")


def restore_defaults() -> dict:
    # downloads + re-embeds 5 papers; needs far more than the default timeout
    return _request("POST", f"{API_URL}/documents/restore-defaults", timeout=600)


def send_feedback(interaction_id: int, rating: int) -> dict:
    return _request("POST", f"{API_URL}/feedback",
                     json={"interaction_id": interaction_id, "rating": rating})


def metrics() -> dict:
    return _request("GET", f"{API_URL}/metrics")


def documents() -> list:
    return _request("GET", f"{API_URL}/documents")
