"""Runs a JSONL answer key through the RAG pipeline and grades it with an LLM judge.

Usage:
    python scripts/run_eval.py <gabarito.jsonl> --docs <pasta-com-pdfs> [--out results.json]

Builds a standalone in-memory index from the PDFs (the app's main index is
not touched), asks every question through RAGService and grades each answer
with a Groq judge against `resposta_esperada`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from rag.config import REWRITE_MODEL
from rag.errors import GenerationError
from rag.evaluation import (completed_ids, doc_hit, judge_messages,
                            load_eval_set, parse_verdict, summarize)
from rag.feedback.db import FeedbackDB
from rag.generation.groq_chat import GroqChat
from rag.ingestion.pipeline import ingest_pdf
from rag.retrieval.embedder import Embedder
from rag.retrieval.reranker import Reranker
from rag.retrieval.store import IndexStore
from rag.service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_file", help="JSONL com pergunta/resposta_esperada")
    parser.add_argument("--docs", required=True, help="pasta com os PDFs da base")
    parser.add_argument("--language", default="pt", choices=["en", "pt"])
    parser.add_argument("--out", default="eval_results.json")
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="pausa entre perguntas (free tier do Groq)")
    parser.add_argument("--resume", action="store_true",
                        help="pula perguntas já presentes no arquivo .partial.jsonl")
    parser.add_argument("--judge-model", default=REWRITE_MODEL,
                        help="modelo do juiz (default: 8b, cota separada do 70b)")
    args = parser.parse_args()

    load_dotenv()
    items = load_eval_set(Path(args.eval_file))
    partial_path = Path(args.out).with_suffix(".partial.jsonl")
    done = completed_ids(partial_path) if args.resume else set()
    if not args.resume and partial_path.exists():
        partial_path.unlink()
    if done:
        print(f"[resume] {len(done)} perguntas já corrigidas; pulando.")
    pdfs = sorted(Path(args.docs).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Nenhum PDF em '{args.docs}'.")

    print("Carregando modelos...")
    embedder = Embedder()
    store = IndexStore()
    for pdf in pdfs:
        added = ingest_pdf(pdf, store, embedder)
        print(f"[index] {pdf.name}: {added} chunks")

    chat = GroqChat()
    unused = Path("data") / "eval_unused"
    service = RAGService(store=store, embedder=embedder, reranker=Reranker(),
                         chat=chat, db=FeedbackDB(":memory:"),
                         index_dir=unused, documents_dir=unused)

    results = [json.loads(line)
               for line in partial_path.read_text(encoding="utf-8").splitlines()
               if line.strip()] if (args.resume and partial_path.exists()) else []

    def _ask_with_quota_wait(fn, what):
        for _ in range(4):
            try:
                return fn()
            except GenerationError as exc:
                message = str(exc).lower()
                if "429" not in message and "rate limit" not in message:
                    raise
                print(f"[quota] limite do Groq atingido em '{what}'; "
                      "aguardando 35 min...", flush=True)
                time.sleep(35 * 60)
        raise GenerationError(f"Cota do Groq seguiu estourada após 4 esperas ({what}).")

    for n, item in enumerate(items, start=1):
        item_id = item.get("id", n)
        if item_id in done:
            continue
        result = _ask_with_quota_wait(
            lambda: service.ask(item["pergunta"], language=args.language),
            f"pergunta {item_id}")
        titles = {s.doc_title for s in result.sources}
        hit = doc_hit(item.get("doc_fonte", "nenhum"), titles)
        verdict_text = _ask_with_quota_wait(
            lambda: chat.complete(
                args.judge_model,
                judge_messages(item["pergunta"], item["resposta_esperada"],
                               result.answer),
                temperature=0.0,
            ),
            f"juiz {item_id}")
        veredito, justificativa = parse_verdict(verdict_text)
        row = {
            "id": item_id, "pergunta": item["pergunta"],
            "tipo": item.get("tipo", "-"), "veredito": veredito,
            "justificativa": justificativa, "doc_hit": hit,
            "resposta": result.answer,
            "resposta_esperada": item["resposta_esperada"],
            "fontes": sorted(titles),
        }
        results.append(row)
        with partial_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        hit_label = {True: "hit", False: "MISS", None: "-"}[hit]
        print(f"[{n}/{len(items)}] id={item_id} {veredito:<10} fonte={hit_label}",
              flush=True)
        time.sleep(args.sleep)

    results.sort(key=lambda r: r["id"])

    summary = summarize(results)
    Path(args.out).write_text(
        json.dumps({"resumo": summary, "resultados": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== RESUMO =====")
    print(f"Acurácia: {summary['acuracia']:.0%} "
          f"({summary['corretos']}/{summary['total']} corretas, "
          f"{summary['parciais']} parciais, {summary['incorretos']} incorretas)")
    if summary["retrieval_hit_rate"] is not None:
        print(f"Recuperação: fonte esperada presente em "
              f"{summary['retrieval_hit_rate']:.0%} das perguntas")
    print("Por tipo:")
    for tipo, stats in sorted(summary["por_tipo"].items()):
        print(f"  {tipo:<18} {stats['corretos']}/{stats['total']}")
    print(f"\nDetalhes em '{args.out}'.")


if __name__ == "__main__":
    main()
