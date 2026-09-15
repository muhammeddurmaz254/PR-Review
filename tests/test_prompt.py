"""The detector's prompt is pinned, and names every defect from the code in front of the reader."""
import hashlib
import re

from prdetect.detect import prompt

# Unpinned only on purpose: a change to the prompt changes what every run means.
PINNED = "af5ff56cd5ed87ad"


def test_the_rendered_prompt_is_pinned():
    actual = hashlib.sha256(prompt.system().encode("utf-8")).hexdigest()
    assert actual.startswith(PINNED), "the prompt changed"
    assert prompt.digest() == actual[:16]


def test_the_catalogue_has_distinct_defined_kinds():
    names = prompt.types()
    assert len(names) == len(set(names)) == 74
    assert list(prompt.definitions()) == names
    assert all(text.strip() for text in prompt.definitions().values())


def test_the_prompt_lists_exactly_the_catalogue():
    assert re.findall(r"^- `([^`]+)` -- ", prompt.system(), re.M) == prompt.types()


def test_the_prompt_describes_how_the_code_is_printed():
    text = prompt.system()
    assert "Lines marked `+`" in text
    assert "`quote` is the code on that line" in text


def test_no_kind_is_a_catch_all_family_name():
    coarse = {"authz", "business_logic", "injection", "race_condition", "data_exposure",
              "secrets", "error_handling", "idempotency"}
    assert not set(prompt.types()) & coarse


# Wording that makes a defect depend on code outside the change: a convention,
# a precedent, a neighbour or a sibling to compare with.
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
