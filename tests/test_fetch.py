"""Turning a Bitbucket pull request into a case, offline."""
from __future__ import annotations

import pytest

from prdetect.bitbucket.client import Bitbucket
from prdetect.bitbucket.fetch import build_case, parse_diff

DIFF = """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -1,3 +1,4 @@
 def pay(amount):
-    check(amount)
+    if amount <= 0:
+        raise ValueError(amount)
     return charge(amount)
diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1 +0,0 @@
-x = 1
"""

PULL = {"id": 7, "title": "Validate payments", "description": None,
        "source": {"commit": {"hash": "abc123def456"}, "branch": {"name": "feature"}},
        "destination": {"commit": {"hash": "0123456789ab"}, "branch": {"name": "main"}},
        "links": {"html": {"href": "https://bitbucket.org/w/r/pull-requests/7"}}}
TREE = {"app.py": "def pay(amount):\n    if amount <= 0:\n        raise ValueError(amount)\n    return charge(amount)\n",
        "lib.py": "def charge(total):\n    return total\n"}
KEY = {"head_commit": "abc123def456", "is_defective": True, "findings": [
    {"type": "null_deref", "file": "app.py", "start_line": 2, "end_line": 3, "in_scope": True, "required": True,
     "title": "t"}]}


def test_the_diff_gives_changed_and_deleted_files_and_added_lines():
    assert parse_diff(DIFF) == (["app.py"], ["old.py"], {"app.py": [2, 3]})


def test_an_unlabelled_pull_request_is_a_case_without_labels():
    row = build_case(PULL, DIFF, TREE, None)
    assert row["case_id"] == "PR-7" and row["labelled"] is False and row["findings"] == []
    assert row["pr_description"] == ""
    assert list(row["head_files"]) == ["app.py"] and list(row["context_files"]) == ["lib.py"]
    assert row["pull_request"]["source_branch"] == "feature"


def test_a_labelled_pull_request_carries_its_labels():
    row = build_case(PULL, DIFF, TREE, KEY)
    assert row["labelled"] and row["is_defective"]
    assert [f["finding_id"] for f in row["findings"]] == ["PR-7-f1"]


def test_an_answer_key_for_another_commit_is_refused():
    with pytest.raises(SystemExit, match="no longer apply"):
        build_case(PULL, DIFF, TREE, {**KEY, "head_commit": "ffffffffffff"})


def test_a_changed_file_missing_from_the_tree_is_refused():
    with pytest.raises(SystemExit, match="not in the tree"):
        build_case(PULL, DIFF, {"lib.py": ""}, None)


def test_credentials_are_required(tmp_path, monkeypatch):
    monkeypatch.delenv("BITBUCKET_CLOUD_EMAIL", raising=False)
    monkeypatch.delenv("BITBUCKET_CLOUD_API_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="missing"):
        Bitbucket.from_env(tmp_path / ".env", cache_dir=None)
    (tmp_path / ".env").write_text("BITBUCKET_CLOUD_EMAIL=a@b.c\nBITBUCKET_CLOUD_API_TOKEN=secret\n")
    assert Bitbucket.from_env(tmp_path / ".env", cache_dir=None).cache_dir is None
