from pathlib import Path

from rag.evaluation import (completed_ids, doc_hit, judge_messages,
                            load_eval_set, parse_verdict, summarize)


def test_completed_ids_missing_file_is_empty(tmp_path):
    assert completed_ids(tmp_path / "nope.jsonl") == set()


def test_completed_ids_reads_ids(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text('{"id": 1, "veredito": "CORRETO"}\n\n{"id": 7, "veredito": "PARCIAL"}\n',
                    encoding="utf-8")
    assert completed_ids(path) == {1, 7}


def test_load_eval_set_skips_blank_lines(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"id": 1, "pergunta": "p1", "resposta_esperada": "r1"}\n'
        "\n"
        '{"id": 2, "pergunta": "p2", "resposta_esperada": "r2"}\n',
        encoding="utf-8",
    )
    items = load_eval_set(path)
    assert [i["id"] for i in items] == [1, 2]


def test_doc_hit_matches_normalized_titles():
    # sources carry prettified titles, the eval file carries pdf filenames
    assert doc_hit("01_regras_do_futebol.pdf", {"01 Regras Do Futebol"}) is True
    assert doc_hit("01_regras_do_futebol.pdf", {"Outro Documento"}) is False


def test_doc_hit_multi_document_any_match():
    fonte = "01_regras_do_futebol.pdf | 05_glossario_e_faq.pdf"
    assert doc_hit(fonte, {"05 Glossario E Faq"}) is True
    assert doc_hit(fonte, {"Nada A Ver"}) is False


def test_doc_hit_nenhum_returns_none():
    assert doc_hit("nenhum", {"01 Regras Do Futebol"}) is None


def test_parse_verdict_well_formed():
    veredito, justificativa = parse_verdict(
        "VEREDITO: CORRETO\nJUSTIFICATIVA: cita os 11 metros."
    )
    assert veredito == "CORRETO"
    assert justificativa == "cita os 11 metros."


def test_parse_verdict_incorreto_not_confused_with_correto():
    veredito, _ = parse_verdict("VEREDITO: INCORRETO\nJUSTIFICATIVA: número errado.")
    assert veredito == "INCORRETO"


def test_parse_verdict_unparseable_returns_indefinido():
    veredito, justificativa = parse_verdict("não sei avaliar")
    assert veredito == "INDEFINIDO"
    assert justificativa == "não sei avaliar"


def test_judge_messages_include_question_and_answers():
    messages = judge_messages("pergunta?", "esperada", "obtida")
    assert messages[0]["role"] == "system"
    joined = " ".join(m["content"] for m in messages)
    assert "pergunta?" in joined and "esperada" in joined and "obtida" in joined


def test_summarize_aggregates_by_verdict_type_and_hits():
    results = [
        {"tipo": "fato_numerico", "veredito": "CORRETO", "doc_hit": True},
        {"tipo": "fato_numerico", "veredito": "INCORRETO", "doc_hit": False},
        {"tipo": "lista", "veredito": "PARCIAL", "doc_hit": True},
        {"tipo": "fora_do_escopo", "veredito": "CORRETO", "doc_hit": None},
    ]
    summary = summarize(results)
    assert summary["total"] == 4
    assert summary["corretos"] == 2
    assert summary["parciais"] == 1
    assert summary["incorretos"] == 1
    assert summary["acuracia"] == 0.5
    assert summary["retrieval_hit_rate"] == 2 / 3  # 'nenhum' fica de fora
    assert summary["por_tipo"]["fato_numerico"] == {"total": 2, "corretos": 1}
