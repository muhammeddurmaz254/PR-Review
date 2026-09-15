"""What the detector is shown: whole files, real line numbers, marks, deletions and facts."""
from __future__ import annotations

import re
from statistics import median

import pytest

from prdetect.cases import is_code
from prdetect.detect import facts, pack
from tests.repositories import NARROW, REPOSITORIES, WIDE, case, cases

# A pull request that deletes lines, and one whose facts block speaks.
DELETES = (WIDE, "PR-28")
HAS_FACTS = (WIDE, "PR-20")


@pytest.fixture(scope="module", params=REPOSITORIES)
def repo(request) -> str:
    return request.param


def _packs(repo: str) -> dict[str, pack.Pack]:
    return {item.case_id: pack.build(item, with_facts=True, with_deletions=True) for item in cases(repo)}


def _numbered_rows(text: str):
    """(file, number, mark, body) for every numbered code row of a pack."""
    current = None
    for row in text.split("\n"):
        if row.startswith("# FILE "):
            current = row[len("# FILE "):].strip()
            continue
        head, _, body = row.partition(" | ")
        number = head[:5].strip()
        if current is not None and number.isdigit():
            yield current, int(number), head[5:].strip(), body


def test_every_pull_request_gets_one_pack_under_one_system_prompt(repo):
    packs = _packs(repo)
    assert set(packs) == {item.case_id for item in cases(repo)}
    assert len({item.system for item in packs.values()}) == 1


def test_line_numbers_and_marks_match_the_file(repo):
    marked = 0
    for item in cases(repo):
        for filename, number, mark, body in _numbered_rows(pack.build(item, with_deletions=True).user):
            assert item.head_files[filename].split("\n")[number - 1] == body, f"{item.case_id}:{number}"
            assert (mark == "+") == (number in item.added_lines.get(filename, frozenset()))
            marked += mark == "+"
    assert marked > 0


def test_the_added_lines_are_the_diffs_added_lines(repo):
    """Read independently off the diff, the new-side `+` lines agree with the case."""
    for item in cases(repo):
        marked: dict[str, set[int]] = {}
        filename, number = "", 0
        for line in item.diff.split("\n"):
            if line.startswith("+++ "):
                filename = line[4:].strip().removeprefix("b/")
                continue
            hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if hunk:
                number = int(hunk.group(1))
                continue
            if line.startswith("+"):
                marked.setdefault(filename, set()).add(number)
                number += 1
            elif line.startswith(" ") or line == "":
                number += 1
        expected = {name: set(lines) for name, lines in item.added_lines.items() if lines}
        assert marked == expected, item.case_id


def test_a_pack_is_empty_only_when_the_change_touches_no_code(repo):
    for item in cases(repo):
        built = pack.build(item)
        assert (built.shown_lines == 0) == (not item.reviewable), item.case_id
        if not item.reviewable:
            assert not item.scored_labels, item.case_id


def test_clean_and_defective_packs_are_about_the_same_size(repo):
    """Length must not give the verdict away."""
    packs = _packs(repo)
    defective = [packs[c.case_id].shown_lines for c in cases(repo) if c.is_defective]
    clean = [packs[c.case_id].shown_lines for c in cases(repo) if not c.is_defective]
    assert 0.5 < median(defective) / median(clean) < 2.0


def test_every_scored_label_is_printed(repo):
    packs = _packs(repo)
    for item in cases(repo):
        rows = {(filename, number) for filename, number, _, _ in _numbered_rows(packs[item.case_id].user)}
        for label in item.scored_labels:
            assert any((label.span.file, n) in rows for n in label.span.lines), label.finding_id


def test_a_cross_file_pull_request_shows_every_changed_file():
    for item in cases(NARROW):
        if any(label.cross_file for label in item.labels):
            text = pack.build(item).user
            for filename in item.head_files:
                assert f"# FILE {filename}" in text, f"{item.case_id} hides {filename}"


def test_only_source_is_printed():
    for item in cases(NARROW):
        text = pack.build(item).user
        for filename in item.head_files:
            assert (f"# FILE {filename}" in text) == is_code(filename)


def test_shown_lines_are_exactly_what_the_pack_printed(repo):
    for item in cases(repo):
        numbered, _ = pack.shown_lines(item)
        for filename, number, _, body in _numbered_rows(pack.build(item).user):
            assert numbered[filename][number] == body, f"{item.case_id}:{number}"


def test_a_line_that_comes_back_is_not_a_removal():
    item = case(*DELETES)
    removed = [text for rows in pack.removed_lines(item).values() for _, text in rows]
    added = {line[1:].strip() for line in item.diff.split("\n") if line.startswith("+") and not line.startswith("+++")}
    assert removed, "this pull request deletes something"
    assert not [text for text in removed if text.strip() in added]


def test_deleted_lines_are_printed_only_when_asked():
    item = case(*DELETES)
    lean = pack.build(item)
    full = pack.build(item, with_deletions=True)
    assert not re.search(r"^\s+- \| ", lean.user, re.M)
    assert re.search(r"^\s+- \| ", full.user, re.M)
    assert pack.DELETIONS_NOTE in full.user
    # A deleted line has no number, so it is not counted as shown code.
    assert full.shown_lines == lean.shown_lines
    assert not pack.shown_lines(item)[1] and pack.shown_lines(item, with_deletions=True)[1]


def test_the_excerpt_carries_the_detectors_line_numbers(repo):
    checked = 0
    for item in cases(repo):
        for label in item.labels:
            numbers = [int(row[:5]) for row in pack.excerpt(item, label.span.file, label.span.start_line)
                       if row[:5].strip()]
            if numbers:
                checked += 1
                assert min(numbers) <= label.span.start_line <= max(numbers), item.case_id
    assert checked


def test_the_excerpt_shows_the_same_deletions_as_the_pack():
    item = case(*DELETES)
    filename, rows = next(iter(pack.removed_lines(item).items()))
    at = rows[0][0]
    lean = pack.excerpt(item, filename, at, radius=12)
    full = pack.excerpt(item, filename, at, radius=12, with_deletions=True)
    assert not [r for r in lean if r[:5].strip() == "" and r.lstrip().startswith("-")]
    assert [r for r in full if r[:5].strip() == "" and r.lstrip().startswith("-")]
    # A removal must not push a numbered line out of the window.
    assert {r[:5].strip() for r in lean} <= {r[:5].strip() for r in full}


def test_facts_are_silent_unless_they_have_something_to_say():
    speaking = [item for item in cases(WIDE) if facts.collect(item)]
    assert [item.case_id for item in speaking] == ["PR-20", "PR-81"]
    assert all(item.is_defective for item in speaking)


def test_facts_only_ask_about_names_the_change_defines():
    for item in cases(WIDE):
        touched = "\n".join(
            item.source_lines(name)[line - 1] for name, lines in item.added_lines.items()
            if name in item.head_files for line in sorted(lines) if line <= len(item.source_lines(name)))
        for fact in facts.collect(item):
            assert fact.subject in touched, fact.subject


def test_the_facts_block_states_answers_not_code():
    item = case(*HAS_FACTS)
    lean = pack.build(item)
    with_facts = pack.build(item, with_facts=True)
    assert "WHAT THE REPOSITORY SAYS" in with_facts.user and "WHAT THE REPOSITORY SAYS" not in lean.user
    assert with_facts.estimated_tokens - lean.estimated_tokens < 200
