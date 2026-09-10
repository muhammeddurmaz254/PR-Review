"""Build the shared product catalogue that every corpus draws its types from.

Section 0.5 of the zincir specification: a corpus that invents its own type
names cannot answer "does the taxonomy travel?", because the two corpora then
share no vocabulary to travel *in*. So the product's rule ids are lifted out of
halka_bench, where they were only ever a per-dataset table, and made a first
class artefact both corpora are subsets of.

Two things are added on the way out:

**A family for every type.** ``schema.FAMILY_BY_TYPE`` knew five of halka's
thirty-nine in-scope types, which is why the ``file+family`` rung scored 2 TP
against 52 FP -- a prediction whose type is missing from the table gets family
``""`` and matches nothing (section 0.3). The family belongs beside the rule id,
in data, not in a literal in the harness.

**Types no corpus has a positive for.** A deployed analyser ships names for
defects a given repository may never contain, and a false alarm under such a
name is still a false alarm (section 0.4). Keeping them in the catalogue is what
makes that measurable; ``metrics.pr_level`` reads the catalogue, not the labels.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HALKA_TAXONOMY = HERE.parents[1] / "halka_bench" / "taxonomy.json"
OUT = HERE / "product_catalog.json"

FAMILIES = {
    "authz": "Who is allowed to reach the object or the mutation.",
    "injection": "Untrusted text reaching an interpreter -- SQL, shell, markup.",
    "ssrf": "The address of an outbound request coming from outside.",
    "deserialization": "Untrusted bytes handed to a loader that can execute.",
    "secrets": "Credential material in the source, the log or the response.",
    "crypto": "Choice and use of cryptographic primitives.",
    "disclosure": "Internal detail leaving the process in a response or a payload.",
    "correctness": "The code does something other than what it says it does.",
    "edge_cases": "The boundary, the empty case, the absent value.",
    "error_handling": "What happens on the failure path.",
    "concurrency": "Ordering, atomicity and repeated delivery.",
    "data_layer": "How stored data is read and written.",
    "resource_perf": "Work done, and handles held, per unit of input.",
    "maintainability": "Structure a later reader has to pay for.",
    "test_quality": "Whether a test can fail for the reason it names.",
    "config": "Values that change behaviour without changing code.",
}

# Family per type. The halka half is a classification of names that already
# existed; the zincir half is declared here because the corpus is built from it.
FAMILY_BY_TYPE = {
    "missing_authz_check": "authz",
    "crossfile_ownership": "authz",
    "sql_injection": "injection",
    "command_injection": "injection",
    "xss": "injection",
    "ssrf_unvalidated_fetch": "ssrf",
    "unsafe_deserialization": "deserialization",
    "hardcoded_credential": "secrets",
    "secret_in_log": "secrets",
    "weak_crypto_primitive": "crypto",
    "error_detail_disclosure": "disclosure",
    "crossfile_data_exposure": "disclosure",
    "wrong_state_check": "correctness",
    "wrong_data_source": "data_layer",
    "silent_overwrite": "data_layer",
    "wrong_argument": "correctness",
    "unvalidated_passthrough": "correctness",
    "crossfile_unit_mismatch": "data_layer",
    "misleading_name": "maintainability",
    "off_by_one": "edge_cases",
    "null_deref": "edge_cases",
    "unguarded_dict_access": "edge_cases",
    "broad_except": "error_handling",
    "swallowed_exception": "error_handling",
    "crossfile_error_propagation": "error_handling",
    "missing_lock": "concurrency",
    "crossfile_idempotency": "concurrency",
    "crossfile_ordering": "concurrency",
    "unreachable_code": "maintainability",
    "unused_symbol": "maintainability",
    "duplicated_block": "maintainability",
    "divergent_change": "maintainability",
    "redundant_work": "resource_perf",
    "work_in_loop": "resource_perf",
    "unclosed_resource": "resource_perf",
    "missing_assertion": "test_quality",
    "hardcoded_endpoint": "test_quality",
    "duplicated_test_block": "test_quality",
    "duplicated_config": "config",
    "removed_dependency": "config",
    "removed_network_config": "config",
}

# The five names zincir_bench needs that halka_bench had no case for. Each is a
# defect class the product can express as one rule id, in the same shape as the
# thirty-five that came before it -- not a rename of an existing name, which is
# what "korpusa ozel ad icat etme" rules out.
NEW_TYPES = {
    "wrong_assertion_target": {
        "rule_ids": ["static.test-quality.wrong-assertion-target"],
        "family": "test_quality",
        "definition": "A test asserts on something other than the effect it names -- the "
                      "input it just built, a mock's own return, or a value the code under "
                      "test never writes -- so it passes whatever that code does.",
    },
    "leaky_test_state": {
        "rule_ids": ["static.test-quality.leaky-state"],
        "family": "test_quality",
        "definition": "A test leaves process state behind, or depends on state another test "
                      "left, so the result depends on which tests ran before it.",
    },
    "unsafe_default": {
        "rule_ids": ["static.config.unsafe-default"],
        "family": "config",
        "definition": "A default value turns a protection off or removes a bound -- "
                      "verification skipped, a timeout unlimited, a debug path left on -- "
                      "where the same setting is explicit elsewhere.",
    },
    "contradictory_setting": {
        "rule_ids": ["static.config.contradictory-setting"],
        "family": "config",
        "definition": "Two settings that are read together state incompatible things, so one "
                      "of them cannot take effect.",
    },
    "removed_config_key": {
        "rule_ids": ["static.config.removed-key"],
        "family": "config",
        "definition": "A configuration key is deleted while code that reads it stays, so the "
                      "reader falls back to a value nobody chose.",
    },
}

ESCAPE_RULE_IDS = ["static.generic.unclassified", "security.generic.unclassified",
                   "crossfile.generic.unclassified"]


def match_class(rule_ids: list[str], owners: dict[str, set[str]]) -> str:
    """bire_bir / paylasimli / karsiliksiz, computed, never written by hand."""
    if not rule_ids:
        return "karsiliksiz"
    if any(len(owners[rule_id]) > 1 for rule_id in rule_ids):
        return "paylasimli"
    return "bire_bir"


def build() -> dict:
    halka = json.loads(HALKA_TAXONOMY.read_text(encoding="utf-8"))
    types: dict[str, dict] = {}
    for name, row in halka["turler"].items():
        family = FAMILY_BY_TYPE.get(name)
        if family is None:
            raise SystemExit(f"{name}: no family declared; the file+family rung would stay broken")
        types[name] = {
            "rule_ids": list(row["kural_kimlikleri"]),
            "family": family,
            "definition": row.get("tanim", ""),
            "corpora": ["halka"] if row.get("korpustaki_vaka_sayisi") else [],
        }
    for name, row in NEW_TYPES.items():
        if name in types:
            raise SystemExit(f"{name}: already in the catalogue; pick the existing name")
        types[name] = {"rule_ids": list(row["rule_ids"]), "family": row["family"],
                       "definition": row["definition"], "corpora": []}

    owners: dict[str, set[str]] = {}
    for name, row in types.items():
        for rule_id in row["rule_ids"]:
            owners.setdefault(rule_id, set()).add(name)
    for name, row in types.items():
        row["match_class"] = match_class(row["rule_ids"], owners)

    return {
        "version": 1,
        "description": (
            "The product's defect vocabulary. Every corpus's taxonomy.json is a subset of "
            "`types`, and no corpus may add a name here without adding it for all of them. "
            "`family` is the source for schema.FAMILY_BY_TYPE; `corpora` records which "
            "corpora have a positive example, and an empty list is deliberate -- a name the "
            "detector can report but no corpus can confirm still costs precision."
        ),
        "families": FAMILIES,
        "escape_rule_ids": {
            "ids": ESCAPE_RULE_IDS,
            "note": "Never a match target. A prediction under one of these is scored as a miss.",
        },
        "rule_ids": sorted(owners),
        "types": dict(sorted(types.items())),
    }


def main() -> int:
    catalog = build()
    OUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in catalog["types"].values():
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    print(f"{len(catalog['types'])} types, {len(catalog['rule_ids'])} rule ids, "
          f"{len(counts)} families -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
