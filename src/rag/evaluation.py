"""Pure helpers for grading a RAG pipeline against a JSONL answer key."""
import json
import re
from pathlib import Path

_VERDICTS = ("INCORRETO", "PARCIAL", "CORRETO")  # longest/most specific first

_JUDGE_SYSTEM = (
    "Você é um corretor rigoroso de um sistema de perguntas e respostas. "
    "Compare a RESPOSTA DO SISTEMA com a RESPOSTA ESPERADA para a pergunta dada. "
    "Julgue apenas o conteúdo factual essencial: números e nomes devem bater; "
    "diferenças de redação não importam. Se a resposta esperada diz que o sistema "
    "deve admitir não saber, considere CORRETO apenas se o sistema de fato não "
    "inventou uma resposta. Responda EXATAMENTE neste formato:\n"
    "VEREDITO: CORRETO ou PARCIAL ou INCORRETO\n"
    "JUSTIFICATIVA: uma frase curta"
)


def load_eval_set(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def completed_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)["id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()}


def _normalize(name: str) -> str:
    return re.sub(r"[\W_]+", "", name.lower())


def doc_hit(doc_fonte: str, source_titles: set[str]) -> bool | None:
    if doc_fonte.strip().lower() in {"", "nenhum"}:
        return None
    expected = {_normalize(Path(part.strip()).stem)
                for part in doc_fonte.split("|")}
    seen = {_normalize(title) for title in source_titles}
    return bool(expected & seen)


def judge_messages(pergunta: str, esperada: str, obtida: str) -> list[dict]:
    user = (f"PERGUNTA: {pergunta}\n"
            f"RESPOSTA ESPERADA: {esperada}\n"
            f"RESPOSTA DO SISTEMA: {obtida}")
    return [{"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user}]


def parse_verdict(text: str) -> tuple[str, str]:
    match = re.search(r"\b(INCORRETO|PARCIAL|CORRETO)\b", text)
    if not match:
        return "INDEFINIDO", text.strip()
    verdict = match.group(1)
    just_match = re.search(r"JUSTIFICATIVA:\s*(.+)", text, re.DOTALL)
    justificativa = (just_match.group(1) if just_match else text).strip()
    return verdict, justificativa


def summarize(results: list[dict]) -> dict:
    total = len(results)
    counts = {v: sum(1 for r in results if r["veredito"] == v)
              for v in ("CORRETO", "PARCIAL", "INCORRETO", "INDEFINIDO")}
    hits = [r["doc_hit"] for r in results if r["doc_hit"] is not None]
    por_tipo: dict[str, dict] = {}
    for r in results:
        stats = por_tipo.setdefault(r["tipo"], {"total": 0, "corretos": 0})
        stats["total"] += 1
        if r["veredito"] == "CORRETO":
            stats["corretos"] += 1
    return {
        "total": total,
        "corretos": counts["CORRETO"],
        "parciais": counts["PARCIAL"],
        "incorretos": counts["INCORRETO"],
        "indefinidos": counts["INDEFINIDO"],
        "acuracia": counts["CORRETO"] / total if total else 0.0,
        "retrieval_hit_rate": (sum(hits) / len(hits)) if hits else None,
        "por_tipo": por_tipo,
    }
