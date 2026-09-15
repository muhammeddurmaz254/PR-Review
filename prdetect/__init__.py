"""prdetect: review a pull request with a local open-weight model and report its defects by file and line.

One command per stage, each run as `python -m prdetect.cli.<stage>`:

    fetch            the repository's Bitbucket pull requests -> data/cases/<repo>.jsonl
    detect           one model call per pull request -> claims
    continue_review  one more call for the changed files the first read left without a claim
    verify           every claim checked by a tool-using verifier; run twice, once with --judge-location
    rules            deterministic checks that need no model
    publish          what both verifiers established, one per site, plus the rules, above a confidence floor
    report           reports/<repo>/<run>.json; report_txt prints it as text
"""
