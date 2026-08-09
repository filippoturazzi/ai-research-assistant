from rag.ingestion.chunker import chunk_pages


def test_short_doc_single_chunk():
    chunks = chunk_pages([(1, "Um parágrafo curto.")], doc_id="doc1", doc_title="Doc 1")
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "doc1:0"
    assert (c.doc_id, c.doc_title, c.page, c.position) == ("doc1", "Doc 1", 1, 0)
    assert "parágrafo curto" in c.text


def test_long_text_splits_with_overlap():
    words = " ".join(f"w{i}" for i in range(30))
    chunks = chunk_pages([(1, words)], doc_id="d", doc_title="D",
                         chunk_words=10, overlap_words=3)
    assert len(chunks) == 3
    # overlap: últimas palavras do chunk 0 reaparecem no início do chunk 1
    tail = chunks[0].text.split()[-3:]
    assert chunks[1].text.split()[:3] == tail


def test_page_attribution():
    chunks = chunk_pages(
        [(1, " ".join(f"a{i}" for i in range(10))), (2, " ".join(f"b{i}" for i in range(10)))],
        doc_id="d", doc_title="D", chunk_words=10, overlap_words=2,
    )
    assert chunks[0].page == 1
    assert chunks[1].page == 2


def test_empty_pages_no_chunks():
    assert chunk_pages([(1, "   ")], doc_id="d", doc_title="D") == []
