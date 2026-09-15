"""Deterministic checks that need no model.

Each rule reads the change itself -- a line it added, a line it deleted -- and
fires on a fact about it:

- `config.off-value`    a bound or a protection set to "off" (`0`, `None`, `False`, ...)
                        in a settings or constants module;
- `config.string-list`  a name that promises several values, given one delimited string;
- `test.weakened`       a test file left with fewer `assert` lines than the change deleted;
- `secret.literal`      a secret written into any file the change touched, example files aside.

Their findings are published only where the model published nothing nearby
(`dedupe.fill_gaps`).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Sequence

from prdetect.cases import Case, is_code
from prdetect.detect import pack as pack_module

# A value that means "off", "unlimited" or "no answer".
OFF = {"0", "-1", "None", '""', "''", "False", "0.0"}

# A name whose value bounds something, or guards something.
BOUND = re.compile(r"(max|min|limit|batch|timeout|retry|retries|ttl|expire|size|count|attempts)", re.I)
GUARD = re.compile(r"(verify|ssl|tls|secure|auth|check|validate|debug|strict)", re.I)

# A name that promises several values.
PLURAL = re.compile(r"(statuses|codes|hosts|keys|origins|methods|users|paths|fields|names|ids)$", re.I)

# A setting is written two ways: a module constant (`MAX_BATCH = 0`) and a field
# on a settings class (`max_rows: int = 0`). The indent cap keeps the rule to
# module and class bodies: a local `retries = 0` inside a function is a counter,
# not a default.
ASSIGNMENT = re.compile(r"^( {0,4})([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*[^=#]+?)?\s*=\s*(.+?)(?:\s*#.*)?$")

# Counted as pytest writes them. A unittest assertion dropped from a test that
# still checks the same thing another way looks exactly like a weakened test,
# and counting cannot tell the two apart.
ASSERT = re.compile(r"\s*assert\b")

CONFIG_NAMES = ("settings.py", "defaults.py", "config.py", "profiles.py", "constants.py")

_UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class Finding:
    case_id: str
    file: str
    line: int
    quote: str
    type: str
    message: str
    rule: str
    # Not a probability: a rule fired or it did not. Set above the publishing
    # confidence floor so the floor never drops a rule finding.
    confidence: float = 0.9


def is_test_file(path: str) -> bool:
    """A test by the naming convention Python test runners use; helpers like `tests/factories.py` are not."""
    name = path.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def is_constants_module(source: str) -> bool:
    """Every top-level statement other than imports and a docstring is an UPPER_CASE assignment."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    body = [node for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
            and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]

    def constant(node: ast.stmt) -> bool:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        return bool(targets) and all(isinstance(t, ast.Name) and _UPPER.match(t.id) for t in targets)

    return len(body) >= 3 and all(constant(node) for node in body)


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
        if not is_code(filename) or not is_config(filename, source):
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

    Whatever reads it tests membership against text, so `50` matches
    `"502,503,504"` and the set it really accepts is not the one written here.
    """
    out = []
    for filename, source in sorted(case.head_files.items()):
        if not is_code(filename) or not is_config(filename, source):
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

    Counted, not judged: more `assert` lines deleted than added in one file. A
    rewrite that replaces its assertions does not fire; dropping one does.
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


SECRET_NAME = re.compile(r"(password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)",
                         re.I)
SECRET_VALUE = re.compile(r"""[:=]\s*["']?([^"'\s]{6,})["']?\s*$""")
PLACEHOLDER = re.compile(r"(change[_-]?me|placeholder|example|xxx+|\.\.\.|<[^>]+>|\$\{|os\.environ|"
                         r"getenv|secrets\.|vault|\*\*\*)", re.I)

# A file whose whole job is to show the shape of a secret without being one.
EXAMPLE_FILE = re.compile(r"(\.example$|\.sample$|\.template$|\.dist$|example\.|sample\.)", re.I)


def hardcoded_secret(case: Case) -> list[Finding]:
    """A secret this change writes down, in any file it touched.

    Not only code: a compose file or a README carries the value just as far.
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


RULES = (unbounded_default, list_typed_as_text, weakened_test, hardcoded_secret)


def run(case: Case, rules: Sequence = RULES) -> list[Finding]:
    out: list[Finding] = []
    for rule in rules:
        out.extend(rule(case))
    return out
