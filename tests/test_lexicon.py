import numpy as np

from t2a.signals.lexicon import LexiconScorer

S = LexiconScorer(frozenset({"GAIN", "STRONG"}), frozenset({"LOSS", "WEAK", "DECLINE"}))


def test_score_is_net_tone_in_minus_one_to_one():
    out = S.score(["Strong gain!", "Weak quarter, big loss", "strong but weak", "no tone words", ""])
    assert np.allclose(out, [1.0, -1.0, 0.0, 0.0, 0.0])


def test_matching_is_case_insensitive_and_whole_word():
    assert S.counts("GAIN gain Gaining") == (2, 0)   # 'Gaining' is a different LM entry
