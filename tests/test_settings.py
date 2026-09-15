"""Settings come from `.env`, the environment overrides them, and a flag overrides both."""
from __future__ import annotations

import pytest

from prdetect import paths, settings
from prdetect.cli import detect
from tests.repositories import NARROW, cases


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text('# model server\nSERVER_URL="http://127.0.0.1:9"\nLLM_MODEL=model-from-env\nEMPTY=\n',
                    encoding="utf-8")
    monkeypatch.setattr(paths, "ENV_FILE", path)
    for name in (settings.SERVER_URL, settings.LLM_MODEL, "EMPTY"):
        monkeypatch.delenv(name, raising=False)
    return path


def test_values_are_read_from_the_env_file(env_file):
    assert settings.get(settings.SERVER_URL) == "http://127.0.0.1:9"
    assert settings.get(settings.LLM_MODEL) == "model-from-env"
    assert settings.get("EMPTY") is None and settings.get("MISSING") is None


def test_the_environment_wins_over_the_env_file(env_file, monkeypatch):
    monkeypatch.setenv(settings.LLM_MODEL, "model-from-shell")
    assert settings.get(settings.LLM_MODEL) == "model-from-shell"


def test_detect_uses_the_server_and_model_from_the_env_file(env_file):
    cases(NARROW)
    with pytest.raises(SystemExit, match=r"127\.0\.0\.1:9"):
        detect.main(["--repo", NARROW, "--limit", "1", "--quiet"])


def test_a_flag_wins_over_the_env_file(env_file):
    cases(NARROW)
    with pytest.raises(SystemExit, match=r"127\.0\.0\.1:8"):
        detect.main(["--repo", NARROW, "--limit", "1", "--base-url", "http://127.0.0.1:8", "--quiet"])
