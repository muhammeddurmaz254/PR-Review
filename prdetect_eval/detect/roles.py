"""What a changed file is, read off the file itself and never off a label.

Three roles can be read that way. A test file, by the naming convention every
Python test runner uses -- `test_*.py`, `*_test.py`, `conftest.py` -- which
leaves helpers like `tests/factories.py` out, since they prove nothing
themselves. A module of constants, where every top-level statement other than
imports and a docstring is an UPPER_CASE assignment: all of them, not a share,
so no threshold was fitted to where the labels sit. Everything else is
application code.

"Configuration" in general is not among them. zincir's `settings.py` is a
function building a dict -- structurally closer to its `deadletter.py` than to
its `defaults.py` -- and no rule tried here caught the one without catching
the other.
"""
from __future__ import annotations

import ast
import re

_UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")

ROLE_NAMES = {"test": "test file", "constants": "module of constants",
              "application": "application code"}


def is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def is_constants_module(source: str) -> bool:
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


def role_of(path: str, source: str) -> str:
    if is_test_file(path):
        return "test"
    if is_constants_module(source):
        return "constants"
    return "application"
