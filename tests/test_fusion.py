from rag.retrieval.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_wins():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 4, 1]], k=60)
    assert fused[0][0] in (1, 2)
    # 2 aparece em pos 1 e 0; 1 em pos 0 e 2 → score(2) = 1/61+1/62 > score(1) = 1/61+1/63
    assert fused[0][0] == 2


def test_scores_formula():
    fused = reciprocal_rank_fusion([[7]], k=60)
    assert fused == [(7, 1 / 61)]


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
