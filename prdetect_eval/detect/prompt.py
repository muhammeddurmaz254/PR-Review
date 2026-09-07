"""Detector prompts, one taxonomy per dataset.

The two corpora do not share a vocabulary. demo_repo names vulnerability classes
in a Flask web application; SWRBench names the kinds of change a reviewer asked
for in eleven open-source libraries. Collapsing them into one list would invent
a correspondence that does not exist, so each keeps its own and the shared
instructions sit around it.

Every type description is written from the corpus's own examples rather than
from a general idea of what the word means, because the metric scores the label
the dataset uses, not a synonym.

The system text for a run is a constant once the taxonomy is fixed, so the whole
run shares a byte-identical prefix and the server reuses its KV cache. A test
asserts nothing per-case leaks into it.
"""
from __future__ import annotations

PROMPT_VERSION = "review/v1"

# A second axis alongside the taxonomy: the two corpora are printed differently,
# and describing the wrong shape is worse than describing none. demo_repo shows
# whole files, SWRBench shows renumbered hunks. Keeping this per dataset also
# keeps demo_repo's system prompt byte-identical to the one its numbers were
# measured under.

INSTRUCTIONS_V1 = """\
You are reviewing one pull request. Find the defects it introduces and name the \
kind of each one.

{format}
## How to look

Read the change against what the title says it does. A defect is a place where \
the new code will behave wrong -- not code that is merely unusual, unfinished or \
differently styled.

Some defects are *removals*: a check, a guard or a filter that used to be there \
and is gone. Those have no `+` line at all, so read the surrounding code and ask \
what it no longer does. Report them on the line left unprotected.

When several files changed, read them together. Some defects only exist in the \
relationship between two files -- a value produced in one unit and consumed in \
another that expects something else -- and each file looks correct alone.

## When not to report

- A control that moved rather than disappeared. If the change deletes a check \
here and adds an equivalent one elsewhere in the same pull request, nothing is \
wrong.
- Code that only looks dangerous. String building with values from a closed set, \
a placeholder in an example file, a broad `except` that re-raises.
- Refactoring with no behaviour change, and removal of code nothing calls.
- Anything you would raise as a preference rather than a defect.

Most pull requests contain no defect. An empty `findings` list is a normal \
answer and a much better one than a guess: a false alarm costs a reviewer's \
trust, a miss costs one line of recall.

## How to answer

Reply with JSON only:

{"findings": [{"file": "...", "line": 0, "type": "...", "title": "...", "confidence": 0.0}]}

- `file` is the path exactly as printed in a FILE header.
- `line` is a line number printed in the code. Point at the line where the \
defect is; anywhere inside the affected function counts, so prefer the most \
specific line you can name.
- `type` must be one of the kinds listed above.
- `title` is one short clause naming the problem -- under twelve words, no \
explanation, no suggested fix.
- `confidence` is between 0 and 1.

Report each defect once.
"""

WHOLE_FILE_FORMAT = """\
You are given the pull request title, and the code it changed with real line \
numbers. Lines marked `+` are the ones this pull request added or rewrote.
"""

HUNK_FORMAT = """\
You are given the pull request title and the parts of each file it changed. The \
rest of every file is not available, so judge what you can see.

Each excerpt is printed under a `# FILE` header naming the path, and the commit \
the excerpt comes from. A row reads `<line> <mark> | <code>`:

- a number and no mark is a line the pull request left alone;
- a number and `+` is a line it added or rewrote;
- `-` and no number is a line it **deleted**. That line is gone from the new \
file, which is why it has no number -- and a deletion is often the defect \
itself, so read those rows as carefully as the added ones.

Numbers are the file's own. When one file appears under two commits its \
numbering restarts from that commit's view, so answer with the numbers printed \
in the excerpt you are pointing at.
"""

FORMATS = {"demo_repo": WHOLE_FILE_FORMAT, "swrbench": HUNK_FORMAT}


# v2 changes one thing: where the precision/recall trade-off is made. It was
# measured and it lost, so the default stays v1; it is kept because the negative
# result is the reason the next stage exists, and a version that cannot be re-run
# is a claim rather than a measurement.
#
# v1 made it inside the model. It asserted a prior ("most pull requests contain
# no defect") and a cost ("a miss costs one line of recall"), and the model
# obeyed -- it answered on ten of fifty SWRBench cases, in a corpus that is half
# defective, and every confidence it returned sat between 0.85 and 0.95. A
# number with no spread is not a control, so `--threshold` had nothing to work
# with and the operating point was whatever the prompt happened to produce.
#
# v2 moves the trade-off out to scoring: report the suspicion and grade it, and
# let the threshold sweep over stored artefacts choose the operating point. The
# claim about cost is also simply false for this tool -- a miss is the whole
# failure, not one line of recall.
#
# Measured on SWRBench: v2 does raise recall (6 located to 7 of 25) and does
# widen the confidence spread (0.85-0.95 to 0.40-0.95). It still loses, because
# the spread carries no signal -- sweeping the threshold on v2 discards true
# positives as fast as false ones, so at every matched false-alarm count v1 is
# ahead (t=0.90: v1 4 hits for 5 alarms, v2 4 for 9). The uncertainty this model
# reports is not calibrated, so the trade-off cannot be moved to a number it
# writes. It has to move to the evidence, or to a second pass.

INSTRUCTIONS_V2 = INSTRUCTIONS_V1.replace(
    """\
Most pull requests contain no defect. An empty `findings` list is a normal \
answer and a much better one than a guess: a false alarm costs a reviewer's \
trust, a miss costs one line of recall.
""",
    """\
Report what you suspect, not only what you can prove, and put how sure you are \
into `confidence`. A doubt you would raise with the author belongs in the list \
at a low confidence rather than outside it; a guess you would not defend at any \
confidence belongs nowhere.

An empty `findings` list is a real answer, and the right one when the change \
does what its title says. It is not the safe default.
""",
).replace(
    "- `confidence` is between 0 and 1.",
    """\
- `confidence` is between 0 and 1, and it is read: around 0.9 when you can \
point at the line and say what breaks, around 0.5 when the code is suspicious \
but the caller, the version or the rest of the file would settle it, around 0.3 \
when it is a question you would ask the author rather than a claim.""",
)


DEMO_REPO_TYPES = {
    "authz": "An ownership or permission check is missing or too weak, so a caller "
             "reaches data or an endpoint that should not be theirs.",
    "business_logic": "The code runs correctly but implements the wrong rule: a value is "
                      "computed, converted or reset in a way that gives a wrong business result.",
    "data_exposure": "A field that should stay internal reaches the client, usually because a "
                     "deny-list or an allow-list stopped covering it.",
    "error_handling": "A failure is swallowed or reported as success, so the caller believes "
                      "something happened that did not.",
    "idempotency": "A repeated or retried request is processed more than once, because the "
                   "key or the duplicate check no longer does its job.",
    "injection": "Untrusted input reaches the text of a query or command instead of being "
                 "passed as a parameter.",
    "race_condition": "Two concurrent requests can interleave and leave inconsistent state, "
                      "typically read-then-write where one atomic operation was needed.",
    "secrets": "A credential is written somewhere it can be read later, such as a log line.",
}

SWRBENCH_TYPES = {
    "F.1 Interface": "An API is used against its contract: the wrong object is passed, an "
                     "attribute is read directly where an accessor was meant, or a signature "
                     "does not match how callers use it.",
    "F.2 Logic": "The logic itself is wrong for some input: a branch that mishandles an edge "
                 "case, a string that is not formatted the way it is meant to be.",
    "F.3 Resource": "Build, packaging, configuration or environment handling is wrong: a path "
                    "or option is parsed loosely, or a dependency is unavailable in some run mode.",
    "F.4 Check": "A validation, assertion or guard that should exist does not, so an invalid "
                 "value flows on unchecked.",
    "F.5 Support": "Code assumes a language or library version it will not always run under, "
                   "so it breaks on the versions the project still supports.",
}

# v3 rewrites the SWRBench catalogue from the twenty-five reviewer rationales
# rather than from the words F.1-F.5 suggest. Measured against v1 and v2: of the
# thirteen findings whose location the model got right, eleven were misnamed and
# all nine misnamings went to `F.2 Logic`. Across every prediction the model said
# `F.2 Logic` thirty-one times and `F.3 Resource` or `F.4 Check` not once, while
# the corpus holds five of each. Four defects in the v1 catalogue explain it:
#
#   * F.3 claimed "a dependency is unavailable in some run mode", which is where
#     the corpus files F.5. The two definitions collided on the same sentence.
#   * F.3's real centre of mass is a name or value that is not there in the form
#     it is used -- an unimported symbol, a parameter that never reaches the
#     query -- and none of that reads as "build, packaging, configuration".
#   * F.1 covered misuse but not the breaking change: two of its five cases are a
#     public alias or class attribute deleted out from under its callers.
#   * F.4 covered the missing guard but not the wrong one: two of its five cases
#     are a guard that exists and asserts something that can be false, or catches
#     so broadly it hides unrelated failures.
#
# Both catalogues stay reachable, because the numbers already published were
# measured under v1's.

SWRBENCH_TYPES_V3 = {
    "F.1 Interface": "Something is used against its contract, or a contract others depend on "
                     "is broken. Both count: a call handed the wrong object or a value of the "
                     "wrong kind for what the callee expects, and a public name -- a "
                     "command-line flag, an alias, a class attribute a subclass overrides -- "
                     "removed or narrowed so existing callers stop working.",
    "F.2 Logic": "The new expression is wrong for input it will actually see: a condition true "
                 "in a case it should not be, a value built from the wrong parts, a format "
                 "string that never interpolates, a statement placed in a scope where it does "
                 "not run when it is needed.",
    "F.3 Resource": "A name, value or setting the code depends on is not available in the form "
                    "it is used: a symbol referenced but never imported, a configuration string "
                    "parsed too loosely, a shared context variable whose type changed under its "
                    "consumers, a parameter computed but never passed on to the call that needs "
                    "it, a build or test entry point that no longer works the documented way.",
    "F.4 Check": "The guarding is wrong. It is missing where an invalid value flows on; or it "
                 "is applied on one path and not on its sibling; or it asserts something that "
                 "can legitimately be false; or it catches so broadly that unrelated failures "
                 "are swallowed.",
    "F.5 Support": "The code assumes a language version, a library version or a dependency that "
                   "will not always be there: syntax or a name valid in only one Python "
                   "version, an import that some supported interpreter lacks, a test needing a "
                   "package this branch does not depend on, or a library feature older releases "
                   "do not have.",
}

TAXONOMIES = {"demo_repo": DEMO_REPO_TYPES, "swrbench": SWRBENCH_TYPES}
TAXONOMIES_V3 = {"demo_repo": DEMO_REPO_TYPES, "swrbench": SWRBENCH_TYPES_V3}

# Two clauses of the shared instructions contradict what the SWRBench labels
# actually are, and both suppress a category the model then never uses.
#
# "removal of code nothing calls" cannot be checked here: no case ships a
# checkout, so the model cannot see a caller and reads every deletion as dead
# code -- including the two F.1 findings that are a deleted flag and a deleted
# class attribute.
#
# "anything you would raise as a preference" throws away F.4 outright. Its
# labels are reviewer comments, and two of them are a reviewer asking for a
# guard that a sibling function already has.

INSTRUCTIONS_V3 = INSTRUCTIONS_V1.replace(
    "- Refactoring with no behaviour change, and removal of code nothing calls.",
    """\
- Refactoring with no behaviour change, and removal of code nothing calls. But \
when you cannot see the callers, a removed *public* name -- a command-line flag, \
an exported alias, a class attribute a subclass would override -- is a breaking \
change and not dead code.""",
).replace(
    "- Anything you would raise as a preference rather than a defect.",
    """\
- Anything you would raise as a preference rather than a defect. A guard is not \
a preference: a check the change applies on one path and not on its sibling, or \
an assertion of something that can legitimately be false, is a defect.""",
)

VERSIONS = {
    "review/v1": (INSTRUCTIONS_V1, TAXONOMIES),
    "review/v2": (INSTRUCTIONS_V2, TAXONOMIES),
    "review/v3": (INSTRUCTIONS_V3, TAXONOMIES_V3),
}


def types(dataset: str) -> list[str]:
    if dataset not in TAXONOMIES:
        raise KeyError(f"unknown dataset {dataset!r}; have {', '.join(TAXONOMIES)}")
    return list(TAXONOMIES[dataset])


def system(dataset: str, version: str = PROMPT_VERSION) -> str:
    """The full system prompt: the shared instructions plus this dataset's kinds."""
    if version not in VERSIONS:
        raise KeyError(f"unknown prompt version {version!r}; have {', '.join(VERSIONS)}")
    instructions, taxonomies = VERSIONS[version]
    catalogue = "\n".join(f"- `{name}` -- {text}" for name, text in taxonomies[dataset].items())
    head, tail = instructions.split("## How to look")
    return (
        f"{head.format(format=FORMATS[dataset])}"
        f"## The kinds of defect you report\n\nReport only these, and nothing else:\n\n"
        f"{catalogue}\n\n"
        f"## How to look{tail}"
    )
