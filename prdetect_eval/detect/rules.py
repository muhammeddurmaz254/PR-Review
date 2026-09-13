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

MEASURED (D25). Family rung, published beside the model's findings through
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

ASSIGNMENT = re.compile(r"([A-Z_][A-Z0-9_]*)\s*=\s*(.+?)(?:\s*#.*)?$")
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
                out.append((number, match.group(1), match.group(2).strip(), lines[number - 1].strip()))
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


RULES = (unbounded_default, list_typed_as_text, weakened_test)


def run(case: Case, rules: Sequence = RULES) -> list[Finding]:
    out: list[Finding] = []
    for rule in rules:
        out.extend(rule(case))
    return out
