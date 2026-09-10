"""The shared product catalogue, and the four measurement faults it closes.

Section 0 of the zincir_bench specification is an audit of the ground the
corpora are measured on. Each test here is one of its findings, written so that
the fault cannot come back silently: every one of them was invisible from the
outside -- the numbers still printed, they were just about something other than
what the column said.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metrics
import schema
from adapters import load_cases
from detect import prompt
from matching import match_all
from schema import Case, Label, Prediction, Span

DATASETS = Path(__file__).resolve().parents[1] / "datasets"
CORPORA = ("halka", "demo_repo", "swrbench")


def _existing(name: str) -> Path:
    path = DATASETS / f"{name}.eval.jsonl"
    if not path.exists():
        pytest.skip(f"{path.name} is not built")
    return path


# --- 0.1 the loader drops nothing the corpus was asked to write --------------

DECLARED_BY_THE_CORPUS = ("grounding", "rule_ids", "product_scope",
                          "product_match_class", "rationale", "file_role")


def test_label_carries_every_field_an_exporter_writes():
    """A field an exporter writes and the loader drops is a field nobody has.

    ``build_halka.py`` emitted five of these from its first commit and
    ``_label`` read none of them, so the grounding distribution and the
    product-facing scope were in the file and out of reach at the same time.
    """
    for field in DECLARED_BY_THE_CORPUS:
        assert field in Label.__dataclass_fields__, field


def test_halka_grounding_survives_the_round_trip():
    cases = load_cases(_existing("halka"))
    labels = [label for case in cases for label in case.labels]
    grounded = [label for label in labels if label.grounding]
    assert len(grounded) == len(labels), "a label reached the harness with no ground"
    assert {label.grounding for label in labels} <= {"a", "b", "c"}
    assert all(label.rule_ids or label.product_match_class == "karsiliksiz"
               for label in labels)


# --- 0.2 the file's role is not the label's role -----------------------------

def test_role_and_file_role_are_separate_axes():
    """``role`` says what the label is in its case; ``file_role`` where it lives.

    halka fills ``role`` with "primary" for every label. Overloading that name
    with application/test/config would have made B3 -- the reason zincir_bench
    exists -- unreadable from the loaded corpus.
    """
    label = Label(finding_id="f", case_id="c", type="unsafe_default", family="config",
                  in_scope=True, required=True, role="primary",
                  spans=(Span("zincir/defaults.py", 3, 3),), file_role="config")
    assert label.role == "primary"
    assert label.file_role == "config"


# --- 0.3 every type resolves a family ----------------------------------------

@pytest.mark.parametrize("dataset", CORPORA)
def test_every_in_scope_type_resolves_a_family(dataset):
    """The ``file+family`` rung matches label family against prediction family.

    A type missing from the table gives the prediction family ``""``, which
    equals no label, so the rung scored 2 TP against 52 FP on halka -- a number
    that reads as a detector failure and was a lookup failure.
    """
    cases = load_cases(_existing(dataset))
    missing = sorted({label.type for case in cases for label in case.in_scope_labels
                      if not schema.FAMILY_BY_TYPE.get(label.type)})
    assert not missing, f"{dataset}: {len(missing)} in-scope types have no family: {missing}"


@pytest.mark.parametrize("dataset", CORPORA)
def test_both_sides_of_a_family_match_use_one_vocabulary(dataset):
    cases = load_cases(_existing(dataset))
    for case in cases:
        for label in case.in_scope_labels:
            assert label.family == Prediction(case.case_id, label.span, label.type).family


def test_the_catalogue_is_the_source_of_the_family_table():
    catalog = json.loads(schema.CATALOG_PATH.read_text(encoding="utf-8"))
    for name, row in catalog["types"].items():
        assert schema.FAMILY_BY_TYPE[name] == row["family"]
        assert row["family"] in catalog["families"]


# --- 0.4 a catalogue name with no positives can still be a false alarm -------

def _clean_case(case_id: str) -> Case:
    return Case(case_id=case_id, pair_id=None, variant="clean", difficulty="hard",
                primary_type=None, is_defective=False, pr_title="", pr_description="",
                changed_files=("zincir/pipeline/worker.py",), noise_files=(), deleted_files=(),
                added_lines={"zincir/pipeline/worker.py": frozenset({1})},
                head_files={"zincir/pipeline/worker.py": "x = 1\n"}, context_files={},
                diff="", labels=(), distractors=())


def test_a_report_under_an_unrepresented_catalogue_type_is_a_false_alarm():
    """The fault: which reports count as a flag came from the loaded labels.

    So a corpus with no ``sql_injection`` case could be told a pull request had
    SQL injection and score the case a true negative. zincir_bench contains
    catalogue types with no positives *by design* -- a closed taxonomy narrower
    than the product's would not be the product's -- which is what made this
    worth fixing before building it.
    """
    absent = "sql_injection"
    case = _clean_case("clean-01")
    cases = [case]
    assert absent not in schema.in_scope_types(cases)
    assert absent in schema.scorable_types(cases)

    prediction = Prediction("clean-01", Span("zincir/pipeline/worker.py", 1, 1), absent)
    results = match_all(cases, [prediction], schema.PRIMARY)
    card = metrics.pr_level(cases, results)
    assert card.false_positive == 1, "a catalogue-name report on a clean case vanished"
    assert card.true_negative == 0


def test_a_report_under_a_name_no_catalogue_knows_still_does_not_count():
    case = _clean_case("clean-02")
    prediction = Prediction("clean-02", Span("zincir/pipeline/worker.py", 1, 1), "vibes")
    results = match_all([case], [prediction], schema.PRIMARY)
    assert metrics.pr_level([case], results).true_negative == 1


# --- 0.5 a prompt variant covers every corpus, or says so --------------------

def test_every_prompt_version_covers_every_registered_dataset():
    """A variant that covers one corpus and falls through for the rest is the
    experiment's own control group, run under the experiment's name."""
    for version, (_, taxonomies) in prompt.VERSIONS.items():
        if version in prompt.OPEN:
            continue
        for dataset in prompt.TAXONOMIES:
            assert dataset in taxonomies, (version, dataset)


def test_an_uncovered_dataset_raises_instead_of_falling_back():
    original = dict(prompt.TAXONOMIES_BROAD)
    prompt.TAXONOMIES_BROAD.pop("zincir")
    try:
        with pytest.raises(KeyError, match="defines no catalogue"):
            prompt.types("zincir", "review/v6-broad")
    finally:
        prompt.TAXONOMIES_BROAD.clear()
        prompt.TAXONOMIES_BROAD.update(original)


def test_the_portability_run_hands_both_corpora_the_same_catalogue():
    halka = prompt.types("halka", "review/v6-shared")
    zincir = prompt.types("zincir", "review/v6-shared")
    assert halka == zincir
    assert set(halka) == schema.catalog_types()
    assert prompt.system("halka", "review/v6-shared") == prompt.system("zincir", "review/v6-shared")


def test_a_corpus_may_only_use_catalogue_names():
    """Section 0.5's rule, checked for every corpus that draws on the catalogue.

    demo_repo and SWRBench predate it and name kinds of change, not kinds of
    defect; merging them would invent a correspondence that does not exist.
    """
    for dataset in ("halka", "zincir"):
        path = DATASETS / f"{dataset}.eval.jsonl"
        if not path.exists():
            continue
        cases = load_cases(path)
        used = {label.type for case in cases for label in case.labels}
        assert used <= schema.catalog_types(), sorted(used - schema.catalog_types())
