"""Layer 2: a reviewer that can go and look instead of guessing.

The single-call reviewer sees the changed files and nothing else. Nine of
the best chain's eleven remaining false alarms were claims whose truth hung
on code it was not shown -- a sibling operation, a caller, what a name means
in another module -- and it answered those by guessing, the same way in two
model families. The broad catalogue's definitions ask for exactly that
comparison ("weaker than its siblings"), so the problem was never the
question; it was that the other side of it was out of view.

So the reviewer gets read-only tools over the repository at this pull
request -- read a file, read it as it was before the change, find where a
name is defined or used, search for text -- and a budget of calls. When it
is done it is asked for its findings under the same constrained schema as
the single-call reviewer, and they go through the same gates, so everything
downstream reads its output unchanged.

The repository comes from the case itself (the changed files plus the
unchanged ones), and the revision before the change is rebuilt by reversing
the diff onto it. Nothing needs git, which is what lets the same code run
on a repository that is not a corpus.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from schema import Case

MAX_LINES = 120
MAX_HITS = 40

TOOL_GUIDE = """\

## Looking things up

You can read the repository at this pull request with tools. Use them when a finding depends on code you were not shown: how a sibling operation does the same thing, what a caller passes, where a name is defined, what a file looked like before the change. A comparison you make has to be one you have looked at. If you look and cannot find the evidence, do not report the finding.

- `read_file(path, start, end)` -- the file as this pull request leaves it, with real line numbers; `+` marks lines it added or rewrote.
- `read_base(path, start, end)` -- the same file as it was before the change.
- `find_definition(name)` -- where a function, class or module-level name is defined.
- `find_usages(name)` -- where a name is used.
- `search(text)` -- lines containing a piece of text.
- `list_files(prefix)` -- the files under a path.

Stop calling tools when you have what you need. You will then be asked for your findings.
"""

FINAL_ASK = ("You are done looking. Give your findings now, as JSON in the answer format described "
             "above. Report only defects this pull request introduces that you established; "
             "an empty list is a normal answer.")

TOOLS = [
    {"type": "function", "function": {"name": name, "description": text, "parameters": {
        "type": "object", "properties": props, "required": required}}}
    for name, text, props, required in (
        ("read_file", "Read a file as this pull request leaves it, with line numbers.",
         {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path"]),
        ("read_base", "Read a file as it was before this pull request.",
         {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path"]),
        ("find_definition", "Where a function, class or module-level name is defined.",
         {"name": {"type": "string"}}, ["name"]),
        ("find_usages", "Where a name is used in the repository.", {"name": {"type": "string"}}, ["name"]),
        ("search", "Lines containing a piece of text.", {"text": {"type": "string"}}, ["text"]),
        ("list_files", "Files whose path starts with a prefix.", {"prefix": {"type": "string"}}, []),
    )
]

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def base_revision(case: Case) -> dict[str, str]:
    """Every file as it was before the change: the diff reversed onto head."""
    head = {**case.context_files, **case.head_files}
    files: dict[str, list] = {}
    old_path = new_path = None
    current = None
    for line in case.diff.split("\n"):
        if line.startswith("diff --git "):
            old_path = new_path = None
            current = None
            continue
        if line.startswith("--- "):
            old_path = None if line[4:].strip() == "/dev/null" else line[4:].strip().removeprefix("a/")
            continue
        if line.startswith("+++ "):
            new_path = None if line[4:].strip() == "/dev/null" else line[4:].strip().removeprefix("b/")
            files[new_path or old_path] = [old_path, new_path, []]
            continue
        match = _HUNK.match(line)
        if match and (new_path or old_path):
            current = (int(match.group(1)), [], [])
            files[new_path or old_path][2].append(current)
            continue
        if current is None or line.startswith("\\"):
            continue
        if line.startswith("+"):
            current[2].append(line[1:])
        elif line.startswith("-"):
            current[1].append(line[1:])
        elif line.startswith(" ") or line == "":
            current[1].append(line[1:])
            current[2].append(line[1:])
    base = dict(head)
    for key, (old, new, hunks) in files.items():
        if new is None:          # deleted by the change
            base[old] = "\n".join(l for _, o, _ in hunks for l in o)
            continue
        if old is None:          # added by the change
            base.pop(new, None)
            continue
        rows, out, at = head.get(new, "").split("\n"), [], 1
        for start, o, n in hunks:
            start = max(start, 1)
            out += rows[at - 1:start - 1] + o
            at = start + len(n)
        base.pop(new, None)
        base[old] = "\n".join(out + rows[at - 1:])
    return base


@dataclass
class Workspace:
    """Read-only views of one pull request's repository."""

    case: Case
    head: dict[str, str] = field(init=False)
    base: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.head = {**self.case.context_files, **self.case.head_files}
        self.base = base_revision(self.case)

    @staticmethod
    def _clip(rows: Sequence[str]) -> str:
        if len(rows) > MAX_HITS:
            rows = list(rows[:MAX_HITS]) + [f"... {len(rows) - MAX_HITS} more not shown"]
        return "\n".join(rows) if rows else "(nothing found)"

    def _read(self, files: dict[str, str], path: str, start: int | None, end: int | None,
              marks: bool) -> str:
        path = str(path).strip().removeprefix("./")
        if path not in files:
            near = [f for f in files if f.endswith(path.split("/")[-1])][:5]
            return f"no such file: {path}" + (f"; did you mean {', '.join(near)}?" if near else "")
        rows = files[path].split("\n")
        lo = max(1, int(start or 1))
        hi = min(len(rows), int(end or lo + MAX_LINES - 1), lo + MAX_LINES - 1)
        added = self.case.added_lines.get(path, frozenset()) if marks else frozenset()
        body = [f"{n:5d} {'+' if n in added else ' '} | {rows[n - 1]}" for n in range(lo, hi + 1)]
        tail = [f"... file continues to line {len(rows)}"] if hi < len(rows) else []
        return "\n".join([f"{path} lines {lo}-{hi} of {len(rows)}"] + body + tail)

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        return self._read(self.head, path, start, end, marks=True)

    def read_base(self, path: str, start: int | None = None, end: int | None = None) -> str:
        if str(path).strip().removeprefix("./") not in self.base:
            return f"{path} did not exist before this pull request"
        return self._read(self.base, path, start, end, marks=False)

    def _trees(self):
        for filename, source in sorted(self.head.items()):
            if not filename.endswith(".py"):
                continue
            try:
                yield filename, source.split("\n"), ast.parse(source)
            except (SyntaxError, ValueError):
                continue

    def find_definition(self, name: str) -> str:
        hits = []
        for filename, rows, tree in self._trees():
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                    hits.append(f"{filename}:{node.lineno}: {rows[node.lineno - 1].strip()}")
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.col_offset == 0:
                        hits.append(f"{filename}:{node.lineno}: {rows[node.lineno - 1].strip()}")
        return self._clip(hits)

    def find_usages(self, name: str) -> str:
        hits = []
        for filename, rows, tree in self._trees():
            seen = set()
            for node in ast.walk(tree):
                used = ((isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load))
                        or (isinstance(node, ast.Attribute) and node.attr == name))
                if used and node.lineno not in seen:
                    seen.add(node.lineno)
                    hits.append(f"{filename}:{node.lineno}: {rows[node.lineno - 1].strip()}")
        return self._clip(hits)

    def search(self, text: str) -> str:
        hits = [f"{f}:{n}: {row.strip()}" for f, source in sorted(self.head.items())
                for n, row in enumerate(source.split("\n"), start=1) if text and text in row]
        return self._clip(hits)

    def list_files(self, prefix: str = "") -> str:
        return self._clip([f for f in sorted(self.head) if f.startswith(str(prefix or ""))])

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        handler: Callable[..., str] | None = {
            "read_file": self.read_file, "read_base": self.read_base, "find_definition": self.find_definition,
            "find_usages": self.find_usages, "search": self.search, "list_files": self.list_files,
        }.get(name)
        if handler is None:
            return f"unknown tool {name!r}"
        try:
            return handler(**{k: v for k, v in (arguments or {}).items()})
        except TypeError as error:
            return f"bad arguments for {name}: {error}"


def review(chat: Callable[..., dict], system: str, user: str, workspace: Workspace, schema: dict,
           max_calls: int = 12) -> tuple[str, list[dict], int]:
    """Run the look-then-answer loop. Returns (final text, transcript, tool calls made)."""
    messages: list[dict] = [{"role": "system", "content": system + TOOL_GUIDE},
                            {"role": "user", "content": user}]
    transcript: list[dict] = []
    calls = 0
    while calls < max_calls:
        reply = chat(messages, tools=TOOLS)
        message = reply.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if reply.get("error") or not tool_calls:
            transcript.append({"assistant": message.get("content", ""), "error": reply.get("error", "")})
            break
        messages.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": tool_calls})
        for tool_call in tool_calls[: max_calls - calls]:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            result = workspace.call(function.get("name", ""), arguments)
            messages.append({"role": "tool", "content": result, "tool_name": function.get("name", "")})
            transcript.append({"tool": function.get("name"), "arguments": arguments,
                               "result": result[:600]})
            calls += 1
    messages.append({"role": "user", "content": FINAL_ASK})
    final = chat(messages, schema=schema)
    transcript.append({"final": (final.get("message") or {}).get("content", ""), "error": final.get("error", "")})
    return (final.get("message") or {}).get("content", ""), transcript, calls
