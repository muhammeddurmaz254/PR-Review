# prdetect-eval

Measurement harness for the PR defect detector, and the detector itself.
Everything except the model call works without a GPU.

Start with `../HANDOFF.md` if you are picking this up. This file is the
reference for the harness's own vocabulary: what a run writes, how a number is
counted, and how to read a report.

## Corpora

Three, none of them in version control — each is generated from a source
directory that is not either. Build before running anything, including the
tests:

```console
python datasets/build_halka.py       # 110 cases, real checkout, the primary one
python datasets/build_swrbench.py    # 50 real OSS pull requests, diff only
python datasets/build_demo_repo.py   # 28 cases, whole files
```

## Layout

| Module | Job |
|---|---|
| `schema.py` | `Span` / `Label` / `Prediction` / `Case`, the cascade rungs, `MatchConfig` |
| `adapters.py` | corpus and prediction-file loaders |
| `matching.py` | one-to-one greedy assignment, closest admissible pair first |
| `metrics.py` | scope-aware scoring, cascade, pull-request level, pairwise, breakdowns, threshold sweep |
| `baselines.py` | the trivial heuristics every real number is reported against |
| `pyunits.py` | line-preserving structural view of a Python file (`ast`) |
| `detect/pack.py` | stage [3] — one pull request per call, real line numbers, `+`/`-` marks |
| `detect/prompt.py` | eight prompt versions, each a (instructions, taxonomy) pair; measured ones are hash-pinned |
| `detect/contract.py` | the response schema, its parser, the per-PR cap and the one-comment-per-line rule |
| `detect/client.py` | stage [4] — Ollama chat, the `silent` stub, model-build reporting |
| `detect/anchor.py` | stage [5] — holds each finding to the line it quotes |
| `detect/challenge.py` | stage [6] — hands each claim back with the lines it is about |
| `report.py` `steps.py` | Markdown and console rendering; `step.md` is the one a person reads |
| `run_detect.py` `run_challenge.py` `run_eval.py` | the three entry points |

There is no stage [2]. Candidate enumeration was measured away: the changed code
fits in the prompt, and selecting sites inside it can only lose findings.

## Commands

```console
python run_detect.py --dataset halka --dry-run          # prompts + budget, no server
python run_detect.py --dataset halka --stub silent      # the precision floor
python run_detect.py --dataset halka --model qwen3.8:27b --base-url https://<ngrok> --no-think
python run_challenge.py --run <run_id> --model qwen3.8:27b --base-url https://<ngrok> --no-think
python run_eval.py --eval datasets/halka.eval.jsonl --predictions runs/<id>/predictions.jsonl
python run_eval.py --eval datasets/halka.eval.jsonl --baseline largest_diff_file
python -m pytest tests/ -q
```

`run_detect.py` writes `predictions.jsonl`; `run_eval.py` scores it. The split
is deliberate — **scoring is a pure function of stored artefacts**, so a changed
metric never costs a second GPU pass.

A run whose calls partly failed exits non-zero and records `complete: false`;
re-run the same command with `--resume` to ask only for what is missing. Resume
refuses if the system prompt or the served model build has changed.

## What a run writes

`runs/<run_id>/` holds `config.json` (model and its digest, context, prompt
version, field order, cap, sampling, harness commit), `packs.jsonl`,
`responses.jsonl`, `predictions.jsonl`, `anchors.jsonl`, `metrics.json`,
`report.md` and `step.md`.

## Prediction format

One JSON object per reported defect:

```json
{"case_id":"authz-01-kusurlu","file":"halka/billing/views.py","line":62,
 "type":"missing_authz_check","confidence":0.9,"detector":"review.llm",
 "stage":"detect","message":"read predicate guards a state-changing endpoint"}
```

`end_line` is optional and defaults to `line`.

## How a number is counted

Every prediction lands in exactly one bucket:

| Bucket | Meaning | Effect |
|---|---|---|
| true positive | matched a **required in-scope** label | precision and recall |
| neutral (optional) | matched an in-scope label the corpus marks `required: false` | reported separately |
| neutral (out of scope) | matched a label outside the current scope | reported separately |
| false alarm | matched nothing | precision |

The neutral buckets are what let a corpus carry real defects nobody labelled as
required: on SWRBench, five were confirmed by hand after a run, and counting
them as recall would have scored the detector against answers its own output
produced.

Matching is one-to-one: a prediction satisfies at most one label and a label
consumes at most one prediction. A label with `equivalent_locations` is
satisfied by its nearest copy.

## Reading a report

- **Pull-request level first.** `step.md` leads with it because it is the
  question the product answers. It is reported as **balanced accuracy**
  (recall + specificity) / 2, which is 0.5 for any strategy that ignores the
  input on any class balance — plain accuracy rewards a lopsided corpus.
- **Cascade** — the drop from `file` to `line` is localization loss, not
  detection loss. On every corpus measured so far it is zero: k=0, k=3 and k=10
  give identical numbers, so there is no line-precision work to do.
- **Pairwise accuracy** — a feature counts only when the defective variant is
  caught *and* its clean twin stays silent. Every trivial baseline scores 0.
- **Trivial baselines** — printed with every run. `hot_spot_memoriser` reads the
  answer key and `every_added_line` reports everything; they are ceilings, not
  competitors. The others are honest and must stay weak.
