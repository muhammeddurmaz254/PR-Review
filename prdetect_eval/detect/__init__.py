"""Stages [3] and [4]: pack context for a candidate site, then ask the model.

The split mirrors the architecture. ``context_pack`` is pure and needs no GPU,
so the prompt budget is measurable before a card is rented; ``client`` is the
only module that talks to a server, and it is interchangeable with a stub so the
whole path is testable offline.

Nothing here scores anything. The output is ``predictions.jsonl``, which
``run_eval.py`` reads exactly as it reads a baseline.
"""
