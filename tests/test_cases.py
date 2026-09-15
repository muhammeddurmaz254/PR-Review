"""Cases, labels and answer keys, and the family table both sides of a match read."""
from __future__ import annotations

import json

import pytest

from prdetect import paths
from prdetect.cases import Prediction, is_code
from prdetect.cli import runs
from prdetect.detect import prompt
from prdetect.scoring.families import family
from tests.repositories import NARROW, REPOSITORIES, WIDE, cases


@pytest.fixture(params=REPOSITORIES)
def repo(request) -> str:
    return request.param


def test_every_catalogue_kind_has_a_family():
    """A kind with no family matches nothing at the family rung."""
    assert [name for name in prompt.types() if not family(name)] == []


def test_a_label_and_a_finding_of_one_kind_share_a_family(repo):
    for item in cases(repo):
        for label in item.labels:
            assert label.family and label.family == Prediction(item.case_id, label.span, label.type).family


def test_the_answer_keys_use_catalogue_names_but_two():
    """Two defects no catalogue name describes: a renewal that throws away the
    remaining term, and a network target dropped from configuration."""
    catalogue = set(prompt.types())
    outside = {(repo, item.case_id, label.type) for repo in REPOSITORIES for item in cases(repo)
               for label in item.labels if label.type not in catalogue}
    assert outside == {(NARROW, "PR-1", "business_logic"), (WIDE, "PR-89", "removed_network_config")}


def test_a_label_is_in_scope_exactly_when_it_is_in_source(repo):
    for item in cases(repo):
        for label in item.labels:
            assert label.in_scope == is_code(label.span.file), label.finding_id


def test_every_scored_label_sits_on_changed_code_or_is_a_deletion(repo):
    for item in cases(repo):
        for label in item.scored_labels:
            assert item.touches(label.span) or label.pure_deletion, label.finding_id
            assert label.title.strip(), label.finding_id


def test_a_cross_file_label_can_be_found_at_either_end():
    cross = [label for item in cases(NARROW) for label in item.labels if label.cross_file]
    assert cross
    for label in cross:
        assert len(label.spans) > 1 and all(span.file != label.span.file for span in label.spans[1:])


def test_every_case_carries_the_repository_behind_the_diff(repo):
    for item in cases(repo):
        assert item.context_files and not set(item.context_files) & set(item.head_files), item.case_id


def test_the_cases_agree_with_their_answer_key(repo):
    path = paths.answer_key_file(repo)
    if not path.exists():
        pytest.skip(f"no answer key for {repo}")
    key = json.loads(path.read_text(encoding="utf-8"))["pull_requests"]
    for item in cases(repo):
        entry = key.get(str(item.pull_request["id"]))
        assert item.labelled == (entry is not None), item.case_id
        if entry:
            assert entry["head_commit"].startswith(item.head_commit)
            assert len(item.labels) == len(entry["findings"]) and item.is_defective == entry["is_defective"]


def test_a_later_stage_reads_the_cases_the_run_used(tmp_path):
    given = tmp_path / "elsewhere.jsonl"
    given.write_text("", encoding="utf-8")
    assert runs.cases_file({"repository": "repo", "cases_file": str(given)}) == given
    assert runs.cases_file({"repository": "repo"}) == paths.cases_file("repo")
    assert runs.cases_file({"repository": "repo", "cases_file": "/gone.jsonl"}) == paths.cases_file("repo")


def test_later_stages_read_the_claims_the_anchor_check_let_through(tmp_path):
    (tmp_path / "predictions.jsonl").write_text("", encoding="utf-8")
    assert runs.claims_file(tmp_path).name == "predictions.jsonl"
    (tmp_path / "predictions.anchored.jsonl").write_text("", encoding="utf-8")
    assert runs.claims_file(tmp_path).name == "predictions.anchored.jsonl"
