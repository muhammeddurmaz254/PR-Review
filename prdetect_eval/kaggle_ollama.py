"""Serve a model from a Kaggle T4 x2 notebook so the local harness can reach it.

Paste one `# %%` block per Kaggle cell. Nothing here is imported by the harness;
the tunnel is the whole interface, and `run_detect.py --base-url` is the client.

Two Kaggle facts drive the setup. The accelerator must be set to **GPU T4 x2**
and **Internet must be on** in the notebook settings sidebar, or the install and
the model pull both fail. And `/kaggle/working` is the small, persisted output
directory -- an 18GB model written there fills it, so `OLLAMA_MODELS` points at
the container disk instead, which is discarded with the session anyway.

Layers are split across both cards by `OLLAMA_SCHED_SPREAD`. The 18GB q4_K_M
build does not fit on one 16GB T4; with the spread it sits across 32GB and
leaves plenty for an 8K context, which is all this corpus asks for.
"""

# %% [cell 1] install Ollama
# Takes about a minute. Requires Internet = on in the notebook settings.
import os
import subprocess
import sys
import time
import urllib.request

# The installer unpacks a zstd archive and the Kaggle image ships without the
# tool, so it fails with "This version requires zstd for extraction". The
# notebook runs as root, so no sudo.
subprocess.run("apt-get update -qq && apt-get install -y -qq zstd", shell=True, check=True)
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
print(subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout)


# %% [cell 2] configure and start the server

MODEL = "qwen3.8:27b"  # 27.3B, q4_K_M, 18GB; see the note at the bottom

os.environ.update({
    "OLLAMA_HOST": "0.0.0.0:11434",
    "OLLAMA_MODELS": "/root/.ollama/models",
    # Hold the weights resident: reloading 18GB between calls would dominate the run.
    "OLLAMA_KEEP_ALIVE": "30m",
    # One request at a time. Parallel slots split the KV cache, and this corpus
    # is 111 short calls, so there is nothing to gain and a context to lose.
    "OLLAMA_NUM_PARALLEL": "1",
    # Use both T4s rather than failing to fit on one.
    "OLLAMA_SCHED_SPREAD": "1",
})

server = subprocess.Popen(
    ["ollama", "serve"],
    stdout=open("/tmp/ollama.log", "wb"), stderr=subprocess.STDOUT, env=os.environ,
)

for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2).read()
        print("ollama is up")
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit(open("/tmp/ollama.log").read()[-4000:])

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)


# %% [cell 3] pull the model
# 18GB plus a 931MB vision encoder; five to fifteen minutes on Kaggle's link.
subprocess.run(["ollama", "pull", MODEL], check=True, env=os.environ)


# %% [cell 4] open the tunnel
#
# First, in the notebook: Add-ons -> Secrets -> Add secret, with
#   Label = NGROK_AUTHTOKEN
#   Value = the token from dashboard.ngrok.com
# then attach it to this notebook. `get_secret` takes the *label*, never the
# token itself -- a token pasted into a cell is saved with the notebook output
# and has to be revoked.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=True)

from kaggle_secrets import UserSecretsClient  # noqa: E402
from pyngrok import conf, ngrok  # noqa: E402

SECRET_LABEL = "NGROK_AUTHTOKEN"
try:
    token = UserSecretsClient().get_secret(SECRET_LABEL)
except Exception as error:
    raise SystemExit(
        f"No Kaggle secret labelled {SECRET_LABEL!r} is attached to this notebook.\n"
        "Add-ons -> Secrets -> Add secret, then tick it for this notebook.\n"
        f"({error})"
    )

conf.get_default().auth_token = token
ngrok.kill()
tunnel = ngrok.connect(11434, "http")
print("BASE URL:", tunnel.public_url)


# %% [cell 5] prove the tunnel serves the API, not an HTML interstitial
import json  # noqa: E402

request = urllib.request.Request(
    f"{tunnel.public_url}/api/chat",
    data=json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with the JSON {\"ok\": true}."}],
        "format": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        "stream": False,
        # Same switch the harness uses. Without it this model reasons first and
        # the reply is no longer the schema-shaped object the parser expects.
        "think": False,
    }).encode(),
    headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"},
)
answer = json.loads(urllib.request.urlopen(request, timeout=600).read())
print(answer["message"]["content"])
print("prompt tokens:", answer.get("prompt_eval_count"), "| eval tokens:", answer.get("eval_count"))


# %% [cell 6] hold the session open while the harness runs locally
# Kaggle stops a notebook that finishes, and the tunnel dies with it. Interrupt
# this cell when the run is done.
while True:
    time.sleep(60)
    print(".", end="", flush=True)


# %% [locally, once cell 5 prints a URL]
#
#   cd prdetect_eval
#   python run_detect.py --model qwen3.8:27b --no-think --base-url https://<id>.ngrok-free.app --limit 2
#   python run_detect.py --model qwen3.8:27b --no-think --base-url https://<id>.ngrok-free.app
#   python run_eval.py --predictions runs/<run_id>/predictions.jsonl
#
# Start with --limit 2 -- four calls that cost a minute and prove the contract
# holds before committing to all 111.
#
# `--no-think` is not optional for this model. Thinking is on by default, and a
# reasoning block is not the schema-shaped object the parser reads, so leaving it
# on sends every response to `rejects.jsonl`. The manifest records the setting,
# so a run is never ambiguous about which mode produced it.
#
# `reasoning_effort` and `preserve_thinking` also exist on this model. They are
# deliberately untouched here: phase 0b measures one configuration, and thinking
# on versus off is a phase 2 ablation with the full set behind it.
