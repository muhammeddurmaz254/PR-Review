# prdetect-eval

Measurement harness for the PR defect detector, and the `authz` detector that
runs against it. Everything except the model call itself works without a GPU.

The corpus lives in `../repo` (fieldops-bench) and is consumed only through
`../repo/eval.jsonl`. Scoring never touches the built git repository, so a
changed metric is re-scored from stored artefacts instead of re-running a
detector — on a GPU pipeline that difference is hours.

## Layout

| Module | Job |
|---|---|
| `schema.py` | `Span` / `Label` / `Prediction` / `Candidate` / `Case`, the cascade rungs, `MatchConfig` |
| `adapters.py` | `eval.jsonl` and prediction-file loaders |
| `matching.py` | one-to-one greedy assignment, closest admissible pair first |
| `metrics.py` | scope-aware scoring, cascade, PR level, pairwise, breakdowns, threshold sweep, stage losses |
| `baselines.py` | the trivial heuristics every real number is reported against |
| `pyunits.py` | line-preserving structural view of a Python file (`ast`, not LibCST) |
| `candidates/` | stage [2] enumerators: arm A population counters, arm B static patterns |
| `detect/context_pack.py` | stage [3] — one file's sites, its sibling table and its code, with real line numbers |
| `detect/prompt.py` | the `authz` system prompt, versioned and byte-stable |
| `detect/contract.py` | the response JSON schema and its parser |
| `detect/client.py` | stage [4] — Ollama chat, plus the `silent` and `flag_all` stubs |
| `report.py` | Markdown and console rendering |
| `run_eval.py` | score a prediction file or a baseline |
| `run_ceiling.py` | oracle-judgment ablation: the ceiling of candidate enumeration |
| `run_detect.py` | run the `authz` detector over the corpus |

## Commands

```console
python run_ceiling.py                                  # phase 0a deliverable
python run_eval.py --baseline largest_diff_file        # one trivial baseline
python run_eval.py --predictions path/to/preds.jsonl   # score a detector run
python -m pytest tests/ -q

python run_detect.py --dry-run                         # prompts + context budget, no server
python run_detect.py --stub flag_all                   # upper bound through the real path
python run_detect.py --model qwen2.5-coder:32b --base-url https://<ngrok>.app
```

`run_detect.py` writes `predictions.jsonl`; `run_eval.py --predictions` scores it.
The split is deliberate — a changed metric never costs a second GPU pass.

## Running the model on a rented card

The detector talks to Ollama over HTTP and nothing else, so a tunnelled remote
card is the same as a local one: start `ollama serve`, expose it, and pass the
URL to `--base-url`. `--model` is checked against `/api/tags` before the corpus
runs, so a wrong tag fails in a second rather than 111 timeouts later.

Decoding is constrained by `detect/contract.RESPONSE_SCHEMA`, and
`detect/prompt.SYSTEM` is a constant with nothing interpolated into it so the
server can reuse its KV cache across calls. Both are asserted by tests.

Every run writes `runs/<run_id>/` holding `config.json` (model, quantization,
context, prompt version, detectors, sampling, harness commit — empty until a
model is in the loop), `predictions.jsonl`, `metrics.json` and `report.md`.

## Prediction format

One JSON object per reported defect:

```json
{"case_id":"authz-001-buggy","file":"apps/workorders/views.py","line":18,
 "type":"authz","confidence":0.85,"detector":"authz.orm_read","stage":"detect",
 "message":"queryset is not scoped to the caller organization"}
```

`end_line` is optional and defaults to `line`.

## How a number is counted

Every prediction lands in exactly one bucket:

| Bucket | Meaning | Effect |
|---|---|---|
| true positive | matched a **required in-scope** label | precision and recall |
| neutral (optional) | matched an in-scope label the corpus marks `required: false` | reported separately |
| neutral (out of scope) | matched a label outside the five current types | reported separately |
| false alarm | matched nothing | precision |

Recall's denominator is the 16 required in-scope labels. Reporting a real defect
the current scope excludes neither helps nor hurts, which is what lets the
corpus keep 27 types while the detector targets 5.

Matching is one-to-one: a prediction satisfies at most one label and a label
consumes at most one prediction. A label with `equivalent_locations` is
satisfied by its nearest copy.

## Reading a report

- **Cascade** — the drop from `file` to `line+/-0` is localization loss, not
  detection loss. They need different fixes.
- **Pairwise accuracy** — a feature counts only when the defective variant is
  caught *and* its clean twin stays silent. Every trivial baseline scores 0 here.
- **Trivial baselines** — printed with every run. `hot_spot_memoriser` reads the
  answer key and `every_added_line` reports everything; they are ceilings, not
  competitors. The other six are honest and must stay weak.
- **Cost** in the ceiling report — coverage bought by widening candidate regions
  shows up as context lines per PR, not as a better ceiling.
