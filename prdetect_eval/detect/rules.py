"""Deterministic checks for the two file kinds the reviewer does not see.

D24 measured the wall: with a config file or a test file alone in front of it,
the model still says nothing in seven of zincir_bench's ten missed label files.
A default of `0` under a comment saying `0 = unlimited`, a list of status codes
typed as `"502,503,504"`, an assertion the change deleted -- each reads as
ordinary code, and asking harder does not help. `continuation.py --mode role`
asked each of these files its own question and was measured to lower the bar
for the file rather than find the defect: it accused the clean twin and the
defective one alike.

So these do not ask. Each rule reads the change itself -- the lines it added,
the lines it deleted -- and fires on a fact, which is the one thing that
separates a defective file from its clean twin: the twin did not make that
change. D21 measured that ruff and pyright say nothing at all about this
class, so nothing else in the pipeline covers it.

Every rule is written to be checkable, not clever. What cannot be established
from our corpora is how well they generalise: each fires exactly once here.
What IS measured is what they cost.

MEASURED (D25, extended in D28). Family rung, published beside the model's findings through
`dedupe`, with the 0.8 confidence floor:

    zincir_dev   8/2/8 -> 11/2/5    F1 0.615 -> 0.759
    halka        42/8/6 unchanged   demo_repo 13/3/4 unchanged
    pooled       63/13/18 -> 66/13/15   F1 0.803 -> 0.825, precision 0.829 ->
                 0.835, recall 0.778 -> 0.815

Three findings, three true, no false alarm anywhere: the rules fire three
times on 164 cases and every firing is a label. Precision and recall both rise
and no corpus moves down, which no change since the deletion channel has done.

Sending them through the verifier instead of publishing them costs one of the
three (zincir 11 -> 10): `retryable-01` comes back `unsettled` because four
tool calls did not reach the consumer that does the membership test. A rule
finding is a fact about the diff -- a line added, a line deleted -- and the
verifier is there to check a claim somebody made up, so they are published
as they are.

The caveat that matters, stated plainly: these rules were written after
reading the cases they catch. Their precision here (3/3) is not evidence they
generalise, only that they are narrow -- they say nothing on 138 cases they
were not written for, including every clean twin of the three they fire on.
D28 added two rules and deleted a third; see each function. The set that ships
is the three above plus `hardcoded_secret`, published through
`dedupe.fill_gaps` so a rule speaks only where the model did not: halka
42/8/6, zincir 11/2/5, demo_repo 13/3/4, pooled F1 0.825 -- unchanged by the
extension, plus one real password in a README that the corpus scores out of
scope and every prompt version missed.

The holdout answered that question (D26) and the answer is good: opened once,
`config.off-value` fired on `HANDLER_TIMEOUT_MS = 0` in `timeout-01-kusurlu`
-- a case it had never seen, a constant it was not written for -- and landed on
the label's own line. It fired nowhere else in the eighteen cases, and on none
of the twelve clean ones. One firing is one firing, but it is the firing of a
rule written on `REPLAY_MAX_BATCH` catching a timeout instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from schema import Case
from . import pack as pack_module
from .roles import is_constants_module, is_test_file

# A value that means "off", "unlimited" or "no answer".
OFF = {"0", "-1", "None", '""', "''", "False", "0.0"}

# A name whose value bounds something, or guards something.
BOUND = re.compile(r"(max|min|limit|batch|timeout|retry|retries|ttl|expire|size|count|attempts)", re.I)
GUARD = re.compile(r"(verify|ssl|tls|secure|auth|check|validate|debug|strict)", re.I)

# A name that promises several values.
PLURAL = re.compile(r"(statuses|codes|hosts|keys|origins|methods|users|paths|fields|names|ids)$", re.I)

# A setting is written two ways in Python: a module constant (`MAX_BATCH = 0`)
# and a field on a settings object (`max_rows: int = 0`, class body). D29's
# first stock_bench run showed the rule knew only the first -- zincir writes
# constants, stock writes a dataclass -- so it read `max_movement_rows: int = 0`
# under a comment saying "0 prints every row" as nothing at all. The indent cap
# keeps it to module and class bodies: a local `retries = 0` inside a function
# is a loop counter, not a default.
ASSIGNMENT = re.compile(r"^( {0,4})([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*[^=#]+?)?\s*=\s*(.+?)(?:\s*#.*)?$")

# Counted as pytest writes it. D29 measured the unittest form too
# (`self.assertEqual(...)`) on stock_bench, a unittest suite, and it lost:
# the one defect it reached, `stock-13`, the model had already named at 0.95,
# so `fill_gaps` passed over the rule; what was left was `stock-13-clean`
# dropping `self.assertIn("GLUE250", text)` from a test that still checks the
# same text -- a legitimate edit with exactly the shape of a weakened test.
# Counting cannot tell those apart, and a heuristic written on that one pair
# would be fitting it. stock 10/3/12 without it, 10/4/12 with it.
ASSERT = re.compile(r"\s*assert\b")

CONFIG_NAMES = ("settings.py", "defaults.py", "config.py", "profiles.py", "constants.py")


@dataclass(frozen=True)
class Finding:
    case_id: str
    file: str
    line: int
    quote: str
    type: str
    message: str
    rule: str
    # Not a probability. A rule either fired or did not, and this says so above
    # the publishing floor: at 0.75 the 0.8 confidence filter silently ate every
    # rule finding, which is how this number was found.
    confidence: float = 0.9


def is_config(filename: str, source: str) -> bool:
    return filename.endswith(CONFIG_NAMES) or is_constants_module(source)


def _added_assignments(case: Case, filename: str) -> list[tuple[int, str, str, str]]:
    """(line, name, value, text) for each assignment on a line this change added."""
    out = []
    lines = case.head_files.get(filename, "").splitlines()
    for number in sorted(case.added_lines.get(filename, frozenset())):
        if 1 <= number <= len(lines):
            match = ASSIGNMENT.match(lines[number - 1])
            if match:
                out.append((number, match.group(2), match.group(3).strip(), lines[number - 1].strip()))
    return out


def unbounded_default(case: Case) -> list[Finding]:
    """A value this change sets to nothing, under a name that bounds or guards."""
    out = []
    for filename, source in sorted(case.head_files.items()):
        if not pack_module.is_code(filename) or not is_config(filename, source):
            continue
        for line, name, value, text in _added_assignments(case, filename):
            if value in OFF and (BOUND.search(name) or GUARD.search(name)):
                what = "a bound" if BOUND.search(name) else "a protection"
                out.append(Finding(case.case_id, filename, line, text, "unsafe_default",
                                   f"`{name}` is set to `{value}`, which removes {what}",
                                   "config.off-value"))
    return out


def list_typed_as_text(case: Case) -> list[Finding]:
    """A name that promises several values, given one delimited string.

    Whatever reads it does a membership test against text, so `50` matches
    `"502,503,504"` and the set it really accepts is not the one written here.
    """
    out = []
    for filename, source in sorted(case.head_files.items()):
        if not pack_module.is_code(filename) or not is_config(filename, source):
            continue
        for line, name, value, text in _added_assignments(case, filename):
            if PLURAL.search(name) and value[:1] in "\"'" and ("," in value or ";" in value):
                out.append(Finding(case.case_id, filename, line, text, "unsafe_default",
                                   f"`{name}` names several values but holds one string, so a "
                                   f"reader testing membership tests substrings",
                                   "config.string-list"))
    return out


def weakened_test(case: Case) -> list[Finding]:
    """A test file this change leaves asserting less than it did.

    Counted, not judged: more `assert` lines deleted than added in one file.
    A rewrite that replaces its assertions does not fire; dropping one does.
    """
    out = []
    removals = pack_module.removed_lines(case)
    for filename, source in sorted(case.head_files.items()):
        if not is_test_file(filename):
            continue
        deleted = [(number, text) for number, text in removals.get(filename, ()) if ASSERT.match(text)]
        if not deleted:
            continue
        lines = source.splitlines()
        added = sum(1 for number in case.added_lines.get(filename, frozenset())
                    if 1 <= number <= len(lines) and ASSERT.match(lines[number - 1]))
        if len(deleted) > added:
            number, text = deleted[0]
            out.append(Finding(case.case_id, filename, number, text.strip(), "missing_assertion",
                               f"this change deletes {len(deleted)} assertion(s) and adds {added}, "
                               f"so the test now passes on less than it did",
                               "test.weakened"))
    return out



# --- What the change took away, where something still reaches for it --------

REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(?:[=<>!~\[].*)?$")
# Only what the application needs to run. Dropping `pytest` from a runtime
# list while the tests still import it is housekeeping, not a defect -- and it
# is what the clean twin of halka's `conf-01` does, which is how this line got
# written.
REQUIREMENT_FILES = ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile")
DICT_KEY = re.compile(r"""["']([a-z][a-z0-9_]{2,})["']\s*:""")
CONST_ASSIGN = re.compile(r"^\s*([A-Z][A-Z0-9_]{2,})\s*=")

# A dependency's import name is not always its distribution name. Only the
# handful that differ and are common enough to matter; anything else is looked
# up as itself.
IMPORT_NAME = {"pyyaml": "yaml", "python-dateutil": "dateutil", "beautifulsoup4": "bs4",
               "pillow": "PIL", "msgpack-python": "msgpack", "attrs": "attr",
               "python-dotenv": "dotenv", "scikit-learn": "sklearn"}


def _repo_text(case: Case, tests: bool = True) -> str:
    """Every line of code this case carries, changed and unchanged.

    `tests=False` leaves the test suite out: a package only the tests import is
    not a runtime dependency, whatever the requirements file it sat in.
    """
    files = {**case.head_files, **(case.context_files or {})}
    return "\n".join(source for name, source in files.items()
                     if tests or not is_test_file(name))


def removed_dependency(case: Case) -> list[Finding]:
    """A package this change drops from the requirements, still imported."""
    out = []
    body = _repo_text(case, tests=False)
    for filename, lines in sorted(pack_module.removed_lines(case).items()):
        if not filename.endswith(REQUIREMENT_FILES):
            continue
        for number, text in lines:
            stripped = text.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = REQUIREMENT.match(stripped)
            if not match:
                continue
            package = match.group(1)
            module = IMPORT_NAME.get(package.lower(), package.replace("-", "_"))
            if re.search(rf"^\s*(?:import\s+{re.escape(module)}\b|from\s+{re.escape(module)}\b)",
                         body, re.M):
                out.append(Finding(case.case_id, filename, number, stripped, "removed_dependency",
                                   f"`{package}` is gone from the requirements while `{module}` is "
                                   f"still imported", "deps.removed"))
    return out


# MEASURED and DELETED: a rule for a config key the change removes while
# something still reads its name. It fired on `dlq-01-kusurlu` AND on
# `dlq-01-temiz`, which delete the identical line and have identical head
# files: the twins differ in what `defaults.STREAMS` contains, which is not in
# the deleted line or in any name a reader mentions. A rule that cannot tell
# the defective change from the clean one is not a rule, and no amount of
# widening the reader search fixes it.


SECRET_NAME = re.compile(r"(password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)",
                         re.I)
SECRET_VALUE = re.compile(r"""[:=]\s*["']?([^"'\s]{6,})["']?\s*$""")
PLACEHOLDER = re.compile(r"(change[_-]?me|placeholder|example|xxx+|\.\.\.|<[^>]+>|\$\{|os\.environ|"
                         r"getenv|secrets\.|vault|\*\*\*)", re.I)


# A file whose whole job is to show the shape of a secret without being one.
EXAMPLE_FILE = re.compile(r"(\.example$|\.sample$|\.template$|\.dist$|example\.|sample\.)", re.I)


def hardcoded_secret(case: Case) -> list[Finding]:
    """A secret this change writes down, in any file it touched.

    Not only code: a compose file, a README or an example carries the same
    value to the same place. `ruff` reads none of these (D21), and neither did
    the model -- demo_repo's README password went unreported by every version.
    """
    out = []
    for filename in sorted(case.head_files):
        if EXAMPLE_FILE.search(filename):
            continue
        lines = case.head_files[filename].splitlines()
        for number in sorted(case.added_lines.get(filename, frozenset())):
            if not 1 <= number <= len(lines):
                continue
            text = lines[number - 1]
            if not SECRET_NAME.search(text):
                continue
            value = SECRET_VALUE.search(text)
            if not value or PLACEHOLDER.search(text):
                continue
            out.append(Finding(case.case_id, filename, number, text.strip(),
                               "hardcoded_credential",
                               "a secret is written into the file rather than read from the "
                               "environment", "secret.literal"))
    return out

# MEASURED (D28) and NOT DEFAULT: `removed_dependency` costs halka one false
# alarm and gains nothing scorable. Its one firing is right -- `defusedxml` is
# gone from `requirements.txt` while the code still imports it -- but the
# corpus files that label under `requirements-dev.txt` and marks it out of
# scope, so the report lands on a different file from the label and counts
# against us. A rule that is correct and unscorable is kept and left off.
RULES = (unbounded_default, list_typed_as_text, weakened_test, hardcoded_secret)


def run(case: Case, rules: Sequence = RULES) -> list[Finding]:
    out: list[Finding] = []
    for rule in rules:
        out.extend(rule(case))
    return out
