"""Settings read from the environment, else from the project's `.env`.

    SERVER_URL                  the Ollama server; an ngrok https URL when the GPU is remote
    LLM_MODEL                   the Ollama model tag, e.g. qwen3.8:27b
    BITBUCKET_CLOUD_EMAIL       the Bitbucket account's email
    BITBUCKET_CLOUD_API_TOKEN   a Bitbucket API token

A variable set in the environment wins over the same name in `.env`, and a flag
given on the command line wins over both.
"""
from __future__ import annotations

import os
from pathlib import Path

from prdetect import paths

SERVER_URL = "SERVER_URL"
LLM_MODEL = "LLM_MODEL"


def read_env_file(path: Path) -> dict[str, str]:
    """`NAME=value` lines; comments and blank lines are skipped, surrounding quotes removed."""
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load(env_file: Path | None = None) -> dict[str, str]:
    return {**read_env_file(env_file or paths.ENV_FILE), **os.environ}


def get(name: str, env_file: Path | None = None) -> str | None:
    """A setting's value, or None when it is unset or empty."""
    return load(env_file).get(name) or None
