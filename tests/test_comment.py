"""Findings as one table comment under a pull request, and the Bitbucket calls that write it."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prdetect.bitbucket import comment
from prdetect.bitbucket.client import Bitbucket, BitbucketError
from prdetect.cli import comment as comment_cli
from prdetect.cli import runs
from prdetect.detect import prompt
from prdetect.severity import ORDER, SEVERITY, severity
from tests.repositories import NARROW, case, cases

FINDINGS = [
    {"case_id": "PR-1", "file": "app/views.py", "line": 30, "type": "missing_lock", "message": "Balance read then written"},
    {"case_id": "PR-1", "file": "app/db.py", "line": 12, "end_line": 14, "type": "sql_injection",
     "message": "Query built | from input\nwith f-string", "assigned_to": {"finding_id": "PR-1-f1"}},
    {"case_id": "PR-1", "file": "app/utils.py", "line": 3, "type": "unused_symbol", "message": "Helper never used"},
]


def test_every_kind_the_prompt_can_name_has_a_severity():
    assert [name for name in prompt.types() if name not in SEVERITY] == []
    assert set(SEVERITY.values()) == set(ORDER)
    assert severity("something_new") == "MEDIUM"


def test_the_comment_is_a_table_most_severe_first():
    raw = comment.render(FINDINGS, "abc123def4567890", "ollama:qwen3.8:27b", "g")
    lines = raw.split("\n")
    assert lines[0] == comment.HEADING
    assert "3 bulgu (1 HIGH, 1 MEDIUM, 1 LOW)." in raw
    rows = [line for line in lines if line.startswith("| ") and line[2].isdigit()]
    assert [row.split(" | ")[1] for row in rows] == ["HIGH", "MEDIUM", "LOW"]
    assert "`app/db.py:12-14`" in rows[0] and "`app/views.py:30`" in rows[1]
    assert "Query built \\| from input with f-string" in rows[0], "a cell stays on one line and one column"
    assert "abc123def456" in raw and "abc123def4567890" not in raw


def test_the_comment_carries_nothing_from_the_answer_key():
    raw = comment.render(FINDINGS, "abc", None, "g")
    assert "PR-1-f1" not in raw and "assigned" not in raw


def test_a_pull_request_without_findings_says_so():
    raw = comment.render([], "abc", None, "g")
    assert raw.startswith(comment.HEADING) and comment.NO_FINDINGS in raw
    assert "|" not in raw and comment.NO_SOURCE not in raw


def test_no_findings_without_source_to_read_is_not_a_clean_review():
    raw = comment.render([], "abc", None, "g", reviews_source=False)
    assert comment.NO_FINDINGS in raw and comment.NO_SOURCE in raw


class FakeComments:
    def __init__(self, existing: list[dict]):
        self.existing = existing
        self.posted: list[str] = []
        self.updated: list[tuple[int, str]] = []

    def comments(self, workspace, repo, pull_id):
        return self.existing

    def post_comment(self, workspace, repo, pull_id, raw):
        self.posted.append(raw)
        return {"id": 99}

    def update_comment(self, workspace, repo, pull_id, comment_id, raw):
        self.updated.append((comment_id, raw))
        return {"id": comment_id}


def _existing(comment_id, raw, **extra):
    return {"id": comment_id, "content": {"raw": raw}, **extra}


def test_a_first_run_creates_the_comment_and_ignores_other_comments():
    others = [_existing(1, "LGTM"), _existing(2, comment.HEADING + "\nold", inline={"path": "a.py"}),
              _existing(3, comment.HEADING + "\nold", deleted=True)]
    client = FakeComments(others)
    assert comment.upsert(client, "w", "r", 7, comment.HEADING + "\nnew") == ("created", 99)
    assert client.posted and not client.updated


def test_a_later_run_updates_the_same_comment():
    client = FakeComments([_existing(5, comment.HEADING + "\nold")])
    assert comment.upsert(client, "w", "r", 7, comment.HEADING + "\nnew") == ("updated", 5)
    assert client.updated == [(5, comment.HEADING + "\nnew")] and not client.posted
    same = FakeComments([_existing(5, comment.HEADING + "\nnew\n")])
    assert comment.upsert(same, "w", "r", 7, comment.HEADING + "\nnew") == ("unchanged", 5)


def test_an_old_table_is_replaced_when_the_findings_are_gone():
    stale = FakeComments([_existing(5, comment.HEADING + "\nold table")])
    raw = comment.render([], "abc", None, "g")
    assert comment.upsert(stale, "w", "r", 7, raw) == ("updated", 5)
    assert stale.updated == [(5, raw)] and not stale.posted


# --- the client, against a fake Bitbucket -------------------------------------------

class FakeBitbucket(BaseHTTPRequestHandler):
    requests: list[tuple[str, str, dict | None]] = []
    fail_post = False
    slow_get_once = False
    slow_post = False

    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, document: dict) -> None:
        body = json.dumps(document).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length)) if length else None

    def do_GET(self) -> None:
        FakeBitbucket.requests.append(("GET", self.path, None))
        if FakeBitbucket.slow_get_once:
            FakeBitbucket.slow_get_once = False
            time.sleep(1.0)
        if "page=2" in self.path:
            self._send(200, {"values": [{"id": 2}]})
        elif self.path.split("?")[0].endswith("/comments"):
            self._send(200, {"values": [{"id": 1}], "next": f"http://{self.headers['Host']}{self.path}&page=2"})
        else:
            self._send(200, {"id": 7, "source": {"commit": {"hash": "abc"}}})

    def do_POST(self) -> None:
        FakeBitbucket.requests.append(("POST", self.path, self._body()))
        if FakeBitbucket.slow_post:
            time.sleep(1.0)
        if FakeBitbucket.fail_post:
            self._send(500, {"error": "boom"})
        else:
            self._send(201, {"id": 42})

    def do_PUT(self) -> None:
        FakeBitbucket.requests.append(("PUT", self.path, self._body()))
        self._send(200, {"id": 42})


@pytest.fixture
def bitbucket():
    FakeBitbucket.requests, FakeBitbucket.fail_post = [], False
    FakeBitbucket.slow_get_once = FakeBitbucket.slow_post = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeBitbucket)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield Bitbucket("a@b.c", "token", cache_dir=None, api=f"http://127.0.0.1:{server.server_port}",
                    timeout=0.3, retry_wait=0.0)
    server.shutdown()


def test_the_client_reads_every_page_of_comments(bitbucket):
    assert [c["id"] for c in bitbucket.comments("w", "r", 7)] == [1, 2]
    assert bitbucket.pull_request("w", "r", 7)["source"]["commit"]["hash"] == "abc"


def test_the_client_writes_a_general_comment_and_updates_it(bitbucket):
    assert bitbucket.post_comment("w", "r", 7, "hello")["id"] == 42
    assert bitbucket.update_comment("w", "r", 7, 42, "again")["id"] == 42
    writes = [r for r in FakeBitbucket.requests if r[0] != "GET"]
    assert writes == [("POST", "/repositories/w/r/pullrequests/7/comments", {"content": {"raw": "hello"}}),
                      ("PUT", "/repositories/w/r/pullrequests/7/comments/42", {"content": {"raw": "again"}})]
    assert "inline" not in writes[0][2], "never attached to a line of code"


def test_a_failed_post_is_not_retried(bitbucket):
    """A POST that failed on the server may already have created its comment."""
    FakeBitbucket.fail_post = True
    with pytest.raises(BitbucketError, match="HTTP 500") as raised:
        bitbucket.post_comment("w", "r", 7, "hello")
    assert raised.value.status == 500
    assert len([r for r in FakeBitbucket.requests if r[0] == "POST"]) == 1


def test_a_read_that_times_out_is_asked_again(bitbucket):
    FakeBitbucket.slow_get_once = True
    assert [c["id"] for c in bitbucket.comments("w", "r", 7)] == [1, 2]
    assert len([r for r in FakeBitbucket.requests if r[0] == "GET"]) == 3, "the slow page, its retry, page 2"


def test_a_post_that_times_out_is_not_asked_again(bitbucket):
    FakeBitbucket.slow_post = True
    with pytest.raises(BitbucketError, match="running the command again finds it") as raised:
        bitbucket.post_comment("w", "r", 7, "hello")
    assert raised.value.status is None
    assert len([r for r in FakeBitbucket.requests if r[0] == "POST"]) == 1


# --- the command ---------------------------------------------------------------------

def _publish_run(tmp_path, rows, **manifest):
    run_dir = tmp_path / "p"
    runs.write_rows(run_dir / "predictions.jsonl", rows)
    runs.write_manifest(run_dir, {"repository": NARROW, "model": "ollama:m", **manifest})
    return run_dir


def test_without_post_nothing_reaches_bitbucket(tmp_path, monkeypatch):
    item = case(NARROW, "PR-14")
    run_dir = _publish_run(tmp_path, [{"case_id": "PR-14", "file": "README.md", "line": 3,
                                        "type": "hardcoded_credential", "message": "Password in README"}])
    monkeypatch.setattr(comment_cli.Bitbucket, "from_env", lambda: pytest.fail("no client without --post"))
    assert comment_cli.main(["--run", str(run_dir), "--quiet"]) == 0
    preview = (run_dir / "comments" / "PR-14.md").read_text(encoding="utf-8")
    assert "| 1 | HIGH | `hardcoded_credential` | `README.md:3` | Password in README |" in preview
    assert item.head_commit[:12] in preview
    assert {row["action"] for row in runs.read_rows(run_dir / "comments.jsonl")} == {"preview"}


def test_post_writes_under_every_pull_request_and_skips_a_moved_branch(tmp_path, monkeypatch):
    moved = case(NARROW, "PR-2")
    heads = {item.pull_request["id"]: item.head_commit for item in cases(NARROW)}
    run_dir = _publish_run(tmp_path, [
        {"case_id": "PR-14", "file": "README.md", "line": 3, "type": "hardcoded_credential", "message": "m"},
        {"case_id": "PR-2", "file": "a.py", "line": 1, "type": "xss", "message": "m"}])

    class Client(FakeComments):
        def pull_request(self, workspace, repo, pull_id):
            head = "ffffffffffff" if pull_id == moved.pull_request["id"] else heads[pull_id]
            return {"source": {"commit": {"hash": head}}}

    client = Client([])
    monkeypatch.setattr(comment_cli.Bitbucket, "from_env", lambda: client)
    code = comment_cli.main(["--run", str(run_dir), "--post", "--case", "PR-2", "--case", "PR-14", "--case", "PR-1",
                             "--quiet"])
    actions = {row["case_id"]: row["action"] for row in runs.read_rows(run_dir / "comments.jsonl")}
    assert actions == {"PR-1": "created", "PR-2": "source moved", "PR-14": "created"}
    assert len(client.posted) == 2 and code == 1, "a skipped pull request is reported"
    assert sum(comment.NO_FINDINGS in raw for raw in client.posted) == 1


def test_an_incomplete_run_does_not_say_there_are_no_findings(tmp_path, monkeypatch):
    heads = {item.pull_request["id"]: item.head_commit for item in cases(NARROW)}
    run_dir = _publish_run(tmp_path, [
        {"case_id": "PR-14", "file": "README.md", "line": 3, "type": "hardcoded_credential", "message": "m"}],
        call_failures=1, complete=False)

    class Client(FakeComments):
        def pull_request(self, workspace, repo, pull_id):
            return {"source": {"commit": {"hash": heads[pull_id]}}}

    client = Client([])
    monkeypatch.setattr(comment_cli.Bitbucket, "from_env", lambda: client)
    code = comment_cli.main(["--run", str(run_dir), "--post", "--case", "PR-1", "--case", "PR-14", "--quiet"])
    actions = {row["case_id"]: row["action"] for row in runs.read_rows(run_dir / "comments.jsonl")}
    assert actions == {"PR-1": "incomplete run", "PR-14": "created"}
    assert len(client.posted) == 1 and comment.NO_FINDINGS not in client.posted[0] and code == 1


def _client_failing_on(pull_to_fail: int, error: BitbucketError):
    heads = {item.pull_request["id"]: item.head_commit for item in cases(NARROW)}

    class Client(FakeComments):
        def pull_request(self, workspace, repo, pull_id):
            return {"source": {"commit": {"hash": heads[pull_id]}}}

        def comments(self, workspace, repo, pull_id):
            if pull_id == pull_to_fail:
                raise error
            return []

    return Client([])


def test_a_pull_request_bitbucket_fails_on_is_recorded_and_the_rest_go_on(tmp_path, monkeypatch):
    run_dir = _publish_run(tmp_path, [
        {"case_id": "PR-1", "file": "a.py", "line": 1, "type": "xss", "message": "m"},
        {"case_id": "PR-14", "file": "README.md", "line": 3, "type": "hardcoded_credential", "message": "m"}])
    client = _client_failing_on(1, BitbucketError("GET x: TimeoutError: The read operation timed out"))
    monkeypatch.setattr(comment_cli.Bitbucket, "from_env", lambda: client)
    code = comment_cli.main(["--run", str(run_dir), "--post", "--case", "PR-1", "--case", "PR-14", "--quiet"])
    rows = {row["case_id"]: row for row in runs.read_rows(run_dir / "comments.jsonl")}
    assert rows["PR-1"]["action"] == "error" and "timed out" in rows["PR-1"]["error"]
    assert rows["PR-14"]["action"] == "created"
    assert code == 1


def test_a_refused_token_stops_the_command(tmp_path, monkeypatch):
    run_dir = _publish_run(tmp_path, [{"case_id": "PR-1", "file": "a.py", "line": 1, "type": "xss", "message": "m"}])
    client = _client_failing_on(1, BitbucketError("GET x: HTTP 403 forbidden", 403))
    monkeypatch.setattr(comment_cli.Bitbucket, "from_env", lambda: client)
    with pytest.raises(BitbucketError, match="403"):
        comment_cli.main(["--run", str(run_dir), "--post", "--case", "PR-1", "--case", "PR-14", "--quiet"])
    assert not client.posted
