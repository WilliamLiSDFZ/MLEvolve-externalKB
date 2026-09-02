"""Analogy retrieval at the improve stage: diagnose the current node's bottleneck, search the
paper corpus for the same problem structure in other subfields, map the mechanism back.

Kept import-light on purpose: `corpus.py` needs only rank_bm25 (+ nltk if present), so the KB
repo's probe script and `utils/replay_analogy.py` can import it without torch or omegaconf.
"""
