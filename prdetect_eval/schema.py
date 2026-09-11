"""Canonical types for the PR defect evaluation harness.

Ground truth and predictions are both reduced to spans over files. Everything
downstream — matching, metrics, baselines, candidate enumeration — speaks these
types and nothing else, so a new prediction source only needs an adapter.

Lines are one-based and ranges include both endpoints, matching the corpus.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Literal

TypeMode = Literal["exact", "family", "none"]

CATALOG_PATH = Path(__file__).resolve().parent / "catalog" / "product_catalog.json"


def _load_catalog() -> dict:
    """The product's shared defect vocabulary, or an empty one if it is absent.

    The catalogue is what makes two corpora comparable at all: both draw their
    type names from it, so "does the taxonomy travel?" is a question about the
    same names in a different repository rather than about two private tables.
    """
    if not CATALOG_PATH.exists():
        return {"types": {}, "families": {}}
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


CATALOG = _load_catalog()


def catalog_types() -> frozenset[str]:
    """Every type the product can name, whether or not a corpus has one.

    A deployed analyser ships names for defects a given repository may never
    contain. Section 0.4: a report under such a name is a false alarm and has to
    be counted as one, which cannot happen while the set of scorable types is
    derived from the loaded labels.
    """
    return frozenset(CATALOG.get("types", {}))


def catalog_family(defect_type: str) -> str:
    row = CATALOG.get("types", {}).get(defect_type)
    return row.get("family", "") if row else ""


def in_scope_types(cases: Iterable["Case"]) -> frozenset[str]:
    """Which defect types the loaded corpus actually scores.

    Read from the data rather than fixed in code: the taxonomy is a property of
    the dataset, and two of them are in use with nothing in common -- one names
    vulnerability classes, the other names change categories.
    """
    return frozenset(label.type for case in cases for label in case.labels if label.in_scope)


def scorable_types(cases: Iterable["Case"], published: Iterable[str] = ()) -> frozenset[str]:
    """The types a report may land under and be counted.

    The corpus's own positives, the whole product catalogue, and every name the
    run itself published. The union, not any one part: SWRBench's change
    categories are not in the catalogue and must still score, and a catalogue
    name with no positive anywhere must still be able to produce a false alarm.

    ``published`` closes the same hole one level up. `review/v6-broad` hands
    halka fifty-three names -- the catalogue's words plus twelve from
    `prompt.BEYOND_CORPUS` that no catalogue carries -- and a report under one
    of those twelve on a clean pull request was scored a true negative. On
    halka-noise that hid `path_traversal` on `inj-02-temiz` and `float_money`
    on `corr-01-temiz`, and put balanced accuracy at 87.2% where it is 85.5%.
    A name the model was offered is a name it can be wrong under. A name
    nothing offered it still does not count.
    """
    return in_scope_types(cases) | catalog_types() | frozenset(published)

FAMILY_BY_TYPE = {
    "authz": "authorization", "authn_bypass": "authorization", "mass_assignment": "authorization",
    "sql_injection": "injection", "command_injection": "injection", "path_traversal": "injection",
    "ssrf": "injection", "unsafe_deserialization": "injection",
    "secrets_exposure": "data_exposure", "sensitive_data_exposure": "data_exposure", "crypto_misuse": "data_exposure",
    "race_condition": "concurrency", "missing_transaction": "concurrency", "non_idempotent_retry": "concurrency",
    "null_deref": "correctness", "inverted_condition": "correctness", "off_by_one": "correctness",
    "exception_swallowing": "correctness", "resource_leak": "correctness", "timezone_bug": "correctness",
    "mutable_default_arg": "correctness",
    "n_plus_one": "data_layer", "bulk_bypass": "data_layer", "migration_risk": "data_layer",
    "dead_code": "maintainability", "duplicate_code": "maintainability", "api_contract_break": "maintainability",
    # demo_repo. Both sides of a match resolve family through this table, so a
    # type missing from it silently makes the family rung unreachable.
    "business_logic": "business_logic",
    "data_exposure": "data_exposure", "secrets": "data_exposure",
    # demo_repo files three cases under the coarse name, beside the precise ones.
    "injection": "injection", "authz": "authorization", "concurrency": "concurrency",
    "error_handling": "correctness", "idempotency": "concurrency",
    # SWRBench keeps its own taxonomy: these name kinds of change, not kinds of
    # vulnerability, and collapsing them into the security families above would
    # invent a correspondence that does not exist.
    "F.1 Interface": "interface", "F.2 Logic": "logic", "F.3 Resource": "resource",
    "F.4 Check": "check", "F.5 Support": "support",
}

# Section 0.3: the table above knew five of halka_bench's thirty-nine in-scope
# types, and a type it does not know resolves to family "" -- which matches
# nothing, so the whole ``file+family`` rung scored 2 TP against 52 FP. The fix
# is not more literals here. The family belongs beside the rule id in the shared
# catalogue, where the corpus that introduces a type also declares its family,
# and the harness reads it. The literals above stay for the two corpora that
# predate the catalogue and name kinds of change rather than kinds of defect.
FAMILY_BY_TYPE.update({name: row["family"] for name, row in CATALOG.get("types", {}).items()
                       if row.get("family")})


@dataclass(frozen=True, order=True)
class Span:
    """A closed line range inside one repository-relative file."""

    file: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError(f"invalid span {self.file}:{self.start_line}-{self.end_line}")

    @property
    def lines(self) -> range:
        return range(self.start_line, self.end_line + 1)

    def distance(self, other: "Span") -> int | None:
        """Line gap to ``other``, or None when the two are in different files.

        Overlapping spans are distance 0, which is what makes a prediction that
        lands anywhere inside a ``block_span`` label exact at k=0.
        """
        if self.file != other.file:
            return None
        return max(0, other.start_line - self.end_line, self.start_line - other.end_line)

    def iou(self, other: "Span") -> float:
        """Overlap of two line ranges, 0.0 when they are in different files.

        Localization is judged by whether a report lands inside the region at
        all, so this is reported beside that verdict rather than instead of it:
        a one-line report inside a twenty-line function is correct localization
        and a low IoU, and both facts are worth seeing.
        """
        if self.file != other.file:
            return 0.0
        overlap = min(self.end_line, other.end_line) - max(self.start_line, other.start_line) + 1
        if overlap <= 0:
            return 0.0
        union = max(self.end_line, other.end_line) - min(self.start_line, other.start_line) + 1
        return overlap / union


@dataclass(frozen=True)
class Label:
    """One ground-truth finding.

    ``spans`` holds the anchor first and any equivalent location after it; a
    prediction at either is equally correct under the ``any_of`` match policy
    that duplicate pairs rely on.
    """

    finding_id: str
    case_id: str
    type: str
    family: str
    in_scope: bool
    required: bool
    role: str
    spans: tuple[Span, ...]
    anchor_rule: str = ""
    anchor_text: str = ""
    in_diff: bool = True
    severity: str = ""
    cwe: str = ""
    # The region a report must land in to count as localized. ``focus`` is the
    # narrower statement of where the defect actually is -- the exact line, or
    # the lines the fix removed -- and only drives the strict IoU column.
    focus: Span | None = None
    title: str = ""
    cross_file: bool = False
    pure_deletion: bool = False
    # --- fields the corpora already wrote and the loader used to drop --------
    #
    # Section 0.1: ``build_halka.py`` has been emitting ``grounding``,
    # ``rule_ids``, ``product_scope``, ``product_match_class`` and ``rationale``
    # since it was written, and ``_label`` read none of them. Anything a corpus
    # is asked to record has to arrive here, or asking for it is theatre.
    #
    # ``file_role`` is section 0.2. ``role`` above is the label's role in its
    # case and already carries "primary"; the role of the *file* the defect
    # lives in -- application, test, config -- is a different axis and needed a
    # different name. It is written by the corpus, never guessed from the path:
    # a defect in ``pipeline/settings.py`` is config by intent, and a defect in
    # ``tests/factories.py`` is not a test defect just because of where it sits.
    file_role: str = ""
    # Which of the three grounds (a: visible in the file's own flow, b: known
    # behaviour of the language or library, c: a convention the base branch
    # establishes in more than one place) makes this a defect at all.
    grounding: str = ""
    rule_ids: tuple[str, ...] = ()
    product_scope: bool = False
    product_match_class: str = ""
    rationale: str = ""

    @property
    def catalog_family(self) -> str:
        """The family the shared catalogue gives this type, "" when unknown."""
        return catalog_family(self.type)

    @property
    def span(self) -> Span:
        return self.spans[0]

    @property
    def scored(self) -> bool:
        """True when missing this label counts as a false negative."""
        return self.in_scope and self.required


@dataclass(frozen=True)
class Distractor:
    """A deliberately tempting but correct location. Never a label."""

    span: Span
    looks_like: str
    why_not: str


@dataclass(frozen=True)
class Prediction:
    """One reported defect. ``detector`` and ``stage`` drive the loss table."""

    case_id: str
    span: Span
    type: str
    confidence: float = 1.0
    detector: str = ""
    stage: str = ""
    message: str = ""

    @property
    def family(self) -> str:
        return FAMILY_BY_TYPE.get(self.type, "")


@dataclass(frozen=True)
class Candidate:
    """A site worth asking the model about. Carries no defect claim.

    ``focus`` is the exact line the enumerator flags; ``region`` is the code the
    model would be shown, normally the enclosing function or class. The two are
    scored separately: the model can localize anywhere inside ``region``, so
    region coverage is the true ceiling, while ``focus`` says how much work the
    model still has to do. An enumerator that widens ``region`` to buy coverage
    pays for it visibly in the context-budget column.
    """

    case_id: str
    type: str
    detector: str
    focus: Span
    region: Span
    symbol: str = ""
    evidence: dict = field(default_factory=dict)

    @property
    def family(self) -> str:
        return FAMILY_BY_TYPE.get(self.type, "")

    def as_prediction(self, confidence: float = 1.0) -> "Prediction":
        """Treat the candidate as if it were a report, for the oracle ablations."""
        return Prediction(
            case_id=self.case_id, span=self.focus, type=self.type,
            confidence=confidence, detector=self.detector, stage="enumerate",
            message=self.symbol,
        )


@dataclass
class Case:
    """One pull request: its narrative, its diff, and its ground truth."""

    case_id: str
    pair_id: str | None
    variant: str
    difficulty: str
    primary_type: str | None
    is_defective: bool
    pr_title: str
    pr_description: str
    changed_files: tuple[str, ...]
    noise_files: tuple[str, ...]
    deleted_files: tuple[str, ...]
    added_lines: dict[str, frozenset[int]]
    head_files: dict[str, str]
    # The rest of the repository at head, when a checkout exists. Two thirds of
    # halka_bench's defects are defined against code the pull request does not
    # touch, so a corpus that ships only the changed files puts most of its own
    # ground truth out of reach. Empty for the diff-only corpora.
    context_files: dict[str, str]
    diff: str
    labels: tuple[Label, ...]
    distractors: tuple[Distractor, ...]
    base_commit: str = ""
    head_commit: str = ""
    branch: str = ""

    @property
    def scored_labels(self) -> tuple[Label, ...]:
        return tuple(label for label in self.labels if label.scored)

    @property
    def in_scope_labels(self) -> tuple[Label, ...]:
        return tuple(label for label in self.labels if label.in_scope)

    def touches(self, span: Span) -> bool:
        """True when the pull request adds or rewrites a line inside ``span``."""
        return bool(self.added_lines.get(span.file, frozenset()).intersection(span.lines))

    @property
    def reviewable(self) -> bool:
        """Whether this pull request changes anything the detector is shown.

        Only source is reviewed, so a change that touches nothing but a manifest
        or a document reaches the model as an empty prompt. Staying quiet on one
        is neither right nor wrong, and counting it as a true negative is the
        same free specificity that made SWRBench's first clean half meaningless.
        """
        from detect.pack import is_code
        return any(is_code(name) for name in self.head_files) or bool(self.diff.strip()
                                                                     and not self.head_files)

    def source_lines(self, filename: str) -> list[str]:
        return self.head_files.get(filename, "").split("\n")


@dataclass(frozen=True)
class MatchConfig:
    """One rung of the localization cascade.

    ``tolerance is None`` drops the line constraint entirely, which is how the
    file and file+type rungs are expressed without a special case.
    """

    tolerance: int | None = 3
    type_mode: TypeMode = "exact"
    name: str = ""

    def label_key(self, label: Label) -> str | None:
        return {"exact": label.type, "family": label.family, "none": None}[self.type_mode]

    def prediction_key(self, prediction: Prediction) -> str | None:
        return {"exact": prediction.type, "family": prediction.family, "none": None}[self.type_mode]

    def label(self) -> str:
        if self.name:
            return self.name
        scope = {"exact": "type", "family": "family", "none": "file"}[self.type_mode]
        return scope if self.tolerance is None else f"{scope}+/-{self.tolerance}"


CASCADE: tuple[MatchConfig, ...] = (
    MatchConfig(tolerance=None, type_mode="none", name="file"),
    MatchConfig(tolerance=None, type_mode="family", name="file+family"),
    MatchConfig(tolerance=None, type_mode="exact", name="file+type"),
    MatchConfig(tolerance=10, type_mode="exact", name="region+/-10"),
    MatchConfig(tolerance=5, type_mode="exact", name="region+/-5"),
    MatchConfig(tolerance=3, type_mode="exact", name="region+/-3"),
    MatchConfig(tolerance=0, type_mode="exact", name="inside region"),
)

# A label's span is the region that contains the defect -- normally the whole
# enclosing function. A report anywhere inside it has localized the defect, and
# ``Span.distance`` is 0 for overlapping ranges, so tolerance 0 states exactly
# that. The looser rungs above measure how far a miss landed from the region.
PRIMARY = MatchConfig(tolerance=0, type_mode="exact", name="inside region")


def with_tolerance(config: MatchConfig, tolerance: int | None) -> MatchConfig:
    return replace(config, tolerance=tolerance, name="")


def spans_of(labels: Iterable[Label]) -> list[Span]:
    return [span for label in labels for span in label.spans]
