"""The family table: both sides of a match resolve a type through it."""

import json

import schema
from corpora import NARROW, WIDE, case_id, cases, corpus
from detect import prompt


def test_every_catalogue_name_resolves_to_a_family():
    """A type with no family resolves to "", which matches nothing: the family
    rung would be blind to exactly the names that reach beyond the corpora."""
    missing = [name for name in prompt.types(WIDE) if not schema.FAMILY_BY_TYPE.get(name)]
    assert missing == []


def test_the_narrow_repositorys_labels_use_the_catalogues_names():
    """demo_repo once labelled with coarse names (`authz`, `business_logic`) the
    catalogue does not carry, so its type rung could not be read beside the other
    corpora. Every label is now filed under a catalogue name, except the one defect
    no catalogue name describes: a renewal that discards the remaining term."""
    catalogue = set(prompt.types(NARROW))
    outside = {(case.case_id, label.type) for case in cases(NARROW)
               for label in case.labels if label.type not in catalogue}
    assert outside == {(case_id(NARROW, "uyelik-suresi-sifirlanmasi"), "business_logic")}


def test_every_cross_file_narrow_label_can_be_pointed_at_from_either_half():
    """A defect that exists only between two files is reported at either end; a
    label with one span scored the other end as a false alarm."""
    path = corpus(NARROW)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cross = [f for row in rows for f in row["findings"] if f["cross_file"]]
    assert cross and all(
        f["equivalent_locations"] and all(eq["file"] != f["file"] for eq in f["equivalent_locations"])
        for f in cross)
