"""The detector's prompt is the measured default, and stays it."""

import hashlib

import pytest

from detect import prompt

# The text under measurement (PLAN D37). Unpinned only by re-measuring.
PINNED = {
    "demo_repo": "af5ff56cd5ed87ad",
    "halka": "af5ff56cd5ed87ad",
    "stock": "af5ff56cd5ed87ad",
    "swrbench": "80166d5d802dee56",
    "zincir": "af5ff56cd5ed87ad"
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
    with pytest.raises(KeyError):
        prompt.system("halka", "review/v15-recall")


# Wording that makes a defect depend on code outside the change: a convention,
# a precedent, a neighbour or sibling to compare with.
CUES = ("sibling", "elsewhere", "other call sites", "convention", "same purpose",
        "every other", "this codebase", "neighbour", "comparable", "already follows",
        "already exist", "where other", "in several places", "explicit list")


def test_no_definition_depends_on_how_the_rest_of_the_repository_does_it():
    """A repository with no sibling code must still have nameable defects.
    Duplication is the one kind whose definition needs the other copy."""
    leaning = [name for name, text in prompt.CATALOGUE.items() if any(cue in text.lower() for cue in CUES)]
    assert leaning == ["duplicated_block", "duplicated_test_block"]


def test_no_instruction_depends_on_how_the_rest_of_the_repository_does_it():
    """The one "elsewhere" left is inside the same pull request: a control that moved."""
    lines = [line for line in prompt.INSTRUCTIONS.splitlines() if any(cue in line.lower() for cue in CUES)]
    assert lines == [line for line in prompt.INSTRUCTIONS.splitlines()
                     if line.startswith("- A control that moved rather than disappeared.")]


def test_the_detector_is_asked_for_its_suspicions():
    assert "checked afterwards by a separate reviewer" in prompt.INSTRUCTIONS
    assert "a miss costs one line of recall" not in prompt.INSTRUCTIONS
