"""The detector's prompt is the measured default, and stays it."""

import hashlib

import pytest

from detect import prompt

# The text measured as the default (PLAN D33-D34). Unpinned only by re-measuring.
PINNED = {
    "demo_repo": "e340819caadc120f",
    "halka": "e340819caadc120f",
    "stock": "e340819caadc120f",
    "swrbench": "2bde991ddb807c8b",
    "zincir": "e340819caadc120f"
}


def test_the_rendered_prompt_is_the_measured_one():
    for dataset, digest in PINNED.items():
        actual = hashlib.sha256(prompt.system(dataset).encode("utf-8")).hexdigest()
        assert actual.startswith(digest), f"{dataset} prompt changed; re-measure or revert"


def test_every_repository_gets_the_same_catalogue():
    lists = [prompt.types(dataset) for dataset in prompt.DATASETS_KNOWN]
    assert all(names == lists[0] for names in lists)
    assert len(lists[0]) == len(set(lists[0])) == 74


def test_a_retired_version_is_refused_not_silently_replaced():
    with pytest.raises(KeyError):
        prompt.system("halka", "review/v6-broad")
    with pytest.raises(KeyError):
        prompt.types("halka", "review/v13-universal")


def test_no_definition_depends_on_how_the_rest_of_the_repository_does_it():
    """A repository with no sibling code must still have nameable defects.
    Duplication is the one kind whose definition needs the other copy."""
    cues = ("sibling", "elsewhere", "other call sites", "convention", "same purpose",
            "every other", "surrounding code", "this codebase")
    leaning = [name for name, text in prompt.CATALOGUE.items() if any(cue in text.lower() for cue in cues)]
    assert leaning == ["duplicated_block"]


def test_the_detector_is_asked_for_its_suspicions():
    assert "checked afterwards by a separate reviewer" in prompt.INSTRUCTIONS
    assert "a miss costs one line of recall" not in prompt.INSTRUCTIONS
