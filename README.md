# prdetect

Reviews a pull request with a local open-weight model and reports the defects it
introduces, by file and line, with a kind from a fixed catalogue. Pull requests
are read from Bitbucket Cloud, and the only thing written back is one comment
under each pull request listing its findings (`comment --post`).

The code under review is confidential, so the model is always a local
open-weight one served by [Ollama](https://ollama.com) — on this machine or on a
rented GPU reached through a tunnel. No hosted model API is used.

## Setup

```console
uv venv --python 3.11 venv
uv pip install --python venv/bin/python -r requirements.txt
```

`.env` at the project root holds the model server and the Bitbucket credentials
(an API token that can read repositories and read and write pull requests):

```
SERVER_URL=https://...          # the Ollama server, e.g. an ngrok tunnel
LLM_MODEL=qwen3.8:27b
BITBUCKET_CLOUD_EMAIL=...
BITBUCKET_CLOUD_API_TOKEN=...
```

A variable set in the shell wins over `.env`, and `--model` / `--base-url` win
over both (`prdetect/settings.py`).

## Pipeline

Every stage is a module under `prdetect.cli`, run from the project root. For one
repository:

```console
python -m prdetect.cli.fetch --repo genis_olcum_reposu
python -m prdetect.cli.detect --repo genis_olcum_reposu --run-id g-detect
python -m prdetect.cli.continue_review --run g-detect
python -m prdetect.cli.verify --run g-detect-cont --run-id g-named
python -m prdetect.cli.verify --run g-detect-cont --run-id g-located --judge-location
python -m prdetect.cli.rules --repo genis_olcum_reposu --run-id g-rules
python -m prdetect.cli.publish --run g-named --agree g-located --rules g-rules --run-id g
python -m prdetect.cli.report --run g
python -m prdetect.cli.report_txt reports/genis_olcum_reposu/g.json
python -m prdetect.cli.comment --run g            # previews in runs/g/comments/
python -m prdetect.cli.comment --run g --post     # writes them under the pull requests
```

| Stage | What it does |
|---|---|
| `fetch` | Every open pull request becomes a case in `data/cases/<repo>.jsonl`: title, description, diff, the changed files and the rest of the repository at the source commit. Labels come from `data/answer_keys/<repo>.json` when it exists. Downloads are cached in `data/cache/`. |
| `detect` | One model call per pull request. The model sees the changed source files whole, with real line numbers, `+` on added lines, deleted lines where they were, and a few counted facts about the names the change defines. It answers in a constrained JSON schema: file, line, kind, title, confidence, and the quoted line. At most 6 claims per pull request; claims on unchanged files are dropped; a quote that is not in what the model was shown drops the claim, and a quote found on another line corrects its number. |
| `continue_review` | Where the first read made a claim but left changed files without one, the same pack is sent again naming what is recorded and what is left. |
| `verify` | Each claim is checked in a fresh context by a verifier with read-only tools over the repository (read a file now or before the change, find a definition or its uses, search). Its cited lines are checked mechanically. Run twice: once with the claim's catalogue name, once (`--judge-location`) with only its sentence. |
| `rules` | Deterministic checks that need no model: a bound or protection set to "off" in configuration, a list written as one string, a test left asserting less, a secret written into a file. |
| `publish` | Claims both verifiers established, one per code site, plus the rules' findings where the model published nothing nearby, above confidence 0.6. |
| `report` | `reports/<repo>/<run>.json`; `report_txt` prints it as text. |
| `comment` | One general comment under each pull request with findings: a table of severity, kind, `file:line` and title, most severe first. Never inline in the code. A later run updates the same comment; a pull request whose branch moved since the fetch is skipped. Without `--post` it only writes previews. |

Severity (HIGH, MEDIUM, LOW) belongs to the kind of defect and is read from
`prdetect/severity.py` after publishing: HIGH is a hole someone can use or a loss
of data, money or integrity, MEDIUM is wrong behaviour at run time, LOW is
maintainability and tests. It does not change what the model is asked.

`detect --dry-run` writes the prompts without a server; `--stub silent` runs the
pipeline with a model that finds nothing. A run whose calls partly failed exits
1; the same command with `--resume` asks only for what is missing (also on
`verify`).

## Runs

Each stage writes `runs/<run-id>/` with a `config.json` manifest (repository,
cases file, model and its build digest, prompt digest, counts, commit) and its
rows as JSON lines. A later stage reads an earlier one by run id.

## Answer keys and scoring

`data/answer_keys/<repo>.json` maps a pull request id to its labels: kind, file,
line range, any equivalent location (the other end of a defect that lives
between two files), whether it is in scope and required, and a title. The key is
never shown to the model; `fetch` refuses a key written for another commit.

A published finding is assigned to a label one-to-one, closest first, in the
same file within 10 lines. The report scores three rungs: **location** (any
kind), **family** (the kinds share a family, `prdetect/scoring/families.py`) and
**type** (the same kind). A finding that matches a label the key does not require,
or one out of scope, is neutral: it counts neither for nor against.

## Layout

| Path | |
|---|---|
| `prdetect/paths.py` | where data, runs and reports live |
| `prdetect/cases.py` | `Case`, `Label`, `Prediction`, `Span`; reading and writing cases and predictions |
| `prdetect/bitbucket/` | `client.py` read-only Bitbucket API; `fetch.py` a pull request into a case |
| `prdetect/detect/` | `prompt` `pack` `facts` `contract` `ollama` `scope` `anchor` `continuation` `agent` `verify` `rules` `dedupe` |
| `prdetect/scoring/` | `families` `matching` `metrics` |
| `prdetect/cli/` | one module per stage; `runs.py` is what they share |
| `tests/` | `python -m pytest` |
| `data/` | `answer_keys/` labels, `cases/` fetched pull requests, `cache/` downloads; none of it is committed |

The detector's and the verifier's prompts are pinned by hash in
`tests/test_prompt.py` and `tests/test_verify.py`. Tests that read a repository
skip until it has been fetched.
