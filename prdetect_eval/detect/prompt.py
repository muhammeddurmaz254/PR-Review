"""The detector's system prompt.

One prompt and one catalogue for every repository. The instructions ask the
detector for every defect it has a real reason to suspect, with an honest
confidence, because every claim is checked afterwards by two verifiers and
only what both establish is published (`run_publish.py --agree`, confidence
floor 0.6). Every definition in the catalogue names a defect from the code in
front of the reader; none depends on how the rest of a repository does things.

Measured as the default in PLAN D33-D34. The rendered text is pinned by
`tests/test_prompt.py`: changing it is a new experiment, not a cleanup.
"""
from __future__ import annotations

PROMPT_VERSION = "review/v15-recall"

WHOLE_FILE_FORMAT = """\
You are given the pull request title, and the code it changed with real line numbers. Lines marked `+` are the ones this pull request added or rewrote.
"""

HUNK_FORMAT = """\
You are given the pull request title and the parts of each file it changed. The rest of every file is not available, so judge what you can see.

Each excerpt is printed under a `# FILE` header naming the path, and the commit the excerpt comes from. A row reads `<line> <mark> | <code>`:

- a number and no mark is a line the pull request left alone;
- a number and `+` is a line it added or rewrote;
- `-` and no number is a line it **deleted**. That line is gone from the new file, which is why it has no number -- and a deletion is often the defect itself, so read those rows as carefully as the added ones.

Numbers are the file's own. When one file appears under two commits its numbering restarts from that commit's view, so answer with the numbers printed in the excerpt you are pointing at.
"""

# How each corpus prints the code it hands the detector.
FORMATS = {
    "demo_repo": WHOLE_FILE_FORMAT,
    "halka": WHOLE_FILE_FORMAT,
    "stock": WHOLE_FILE_FORMAT,
    "swrbench": HUNK_FORMAT,
    "zincir": WHOLE_FILE_FORMAT,
}

DATASETS_KNOWN = tuple(sorted(FORMATS))

INSTRUCTIONS = """\
You are reviewing one pull request. Find the defects it introduces and name the kind of each one.

{format}
## How to look

Read the change against what the title says it does.

A defect is a place this change makes worse and a reviewer would ask to fix. Most are about behaviour: the new code will do the wrong thing when it runs. Not all of them are, and these count too:

- it will behave wrong *later*, when something else changes -- a value copied away from the single source that documents it, a block duplicated so that one contract now has to be remembered in two places;
- it does the right thing wastefully -- the same expensive call twice in one flow, or one query per row where a batch call already exists;
- it is named against the repository's own convention, so callers read it wrongly even though it runs correctly;
- it is in configuration, packaging or a dependency list rather than in code -- a package moved out of the runtime requirements, a host dropped from an allow-list the code still reaches for.

Formatting and taste are still not defects, and neither is code that is merely unusual or unfinished.

Some defects are *removals*: a check, a guard or a filter that used to be there and is gone. Those have no `+` line at all, so read the surrounding code and ask what it no longer does. Report them on the line left unprotected.

When several files changed, read them together. Some defects only exist in the relationship between two files -- a value produced in one unit and consumed in another that expects something else -- and each file looks correct alone.

Tests, fixtures, examples and packaging are part of the change and defects in them count. A test is not exempt for being a test:

- it cannot run where the project still runs -- it needs a package this branch does not depend on, or a library feature older versions lack;
- it never really runs -- a skip condition that is always true, a fixture that returns before it asserts, a parametrisation with no values;
- it passes on the wrong thing -- it asserts against a value it also computed, or checks something other than what it names;
- a fixture or `conftest` guard swallows failures the tests were meant to show.

The same holds for an example script and for `setup.py`: an example that misuses the API teaches the misuse, and a packaging change that breaks the documented way of running the tests breaks it for everyone.

## When not to report

- A control that moved rather than disappeared. If the change deletes a check here and adds an equivalent one elsewhere in the same pull request, nothing is wrong.
- Code that only looks dangerous. String building with values from a closed set, a placeholder value standing in for real input, an `except` that re-raises. This is about the expression, not the file it is in -- and an `except` that swallows rather than re-raises is a defect, not a false alarm.
- Refactoring that leaves one copy of the logic, and removal of code nothing calls. Duplicating a block is not refactoring. But when you cannot see the callers, a removed *public* name -- a command-line flag, an exported alias, a class attribute a subclass would override -- is a breaking change and not dead code.
- Anything you would raise as a preference rather than a defect -- but a rule this repository already follows in several places is not a preference, it is a contract. A guard is not a preference: a check the change applies on one path and not on its sibling, or an assertion of something that can legitimately be false, is a defect.

Report anything you have a real reason to suspect, not only what you are sure of. What you report is checked afterwards by a separate reviewer that reads the rest of the repository and drops what it cannot establish, so a suspicion that turns out to be wrong is cheap here, while a defect you keep to yourself is gone for good. Say how sure you are in `confidence` and let the check do its work.

Still report nothing you cannot point at: a line, a quote, and what is wrong with it. An empty `findings` list is the right answer for a pull request where you can find nothing to point at.

## How to answer

Reply with JSON only:

{"findings": [{"file": "...", "line": 0, "quote": "...", "type": "...", "title": "...", "confidence": 0.0}]}

- `file` is the path exactly as printed in a FILE header.
- `line` is a line number printed in the code. Point at the line where the defect is; anywhere inside the affected function counts, so prefer the most specific line you can name.
- `quote` is the code on that line, copied exactly as printed -- without the line number or the `+`/`-` mark, and without the surrounding lines. Copy it; do not retype it from memory. If the defect is a line the change *deleted*, quote that deleted line.
- `type` must be one of the kinds listed above.
- `title` is one short clause naming the problem -- under twelve words, no explanation, no suggested fix.
- `confidence` is between 0 and 1.

Report each defect once.
"""

CATALOGUE = {
    "missing_authz_check": "An operation that reads or changes someone's data runs without a check that this caller may touch that particular record.",
    "crossfile_ownership": "A query selects rows by identifier alone, with no tenant or owner condition, so a caller can name a row that is not theirs.",
    "crossfile_data_exposure": "Serialization emits the object whole, including fields that carry credentials, internal identifiers or personal data.",
    "crossfile_error_propagation": "The callee reports a failure and the caller reports success; the two disagree about what happened.",
    "crossfile_idempotency": "The key the duplicate guard is keyed on is not stable across runs, so the guard it feeds can never match.",
    "crossfile_ordering": "The consumer depends on an order the producer no longer establishes.",
    "crossfile_unit_mismatch": "The value passed is in a different unit from the one the callee's name, signature or docstring declares.",
    "sql_injection": "A runtime value is placed in the query text rather than in the parameters the driver binds.",
    "command_injection": "A runtime value is placed in the command text rather than in an element of an argument list.",
    "xss": "A value that came from outside reaches the response as markup, unescaped.",
    "error_detail_disclosure": "The response carries exception text, a stack trace or an internal identifier back to the caller.",
    "hardcoded_credential": "A secret is a literal here where its neighbours read the same kind of value from configuration.",
    "weak_crypto_primitive": "The primitive chosen is broken for the job it is doing: a digest that is not collision-resistant used for signing or passwords, or a secret compared with an equality that returns early.",
    "unvalidated_passthrough": "Request data is passed straight into a call that acts on it -- a query, a path, a command, a write -- with no check on the way.",
    "wrong_argument": "The arguments do not match the order or the roles the callee's signature declares.",
    "wrong_data_source": "The value is read from a replica, a cache or a copy that can lag, where the decision made from it needs the current one.",
    "wrong_state_check": "A state is compared against a value nothing in the code ever assigns to it.",
    "silent_overwrite": "A write replaces a structure whose other writers merge into it.",
    "unreachable_code": "The control flow above these statements always leaves before them.",
    "unused_symbol": "The symbol is defined and nothing that can see it refers to it.",
    "misleading_name": "The name says something the body does not do -- a `get_` that writes, a `validate_` that returns without deciding anything.",
    "duplicated_block": "The same logic already exists elsewhere and this is a second copy rather than a use of it.",
    "duplicated_config": "The value is defined a second time, so the two definitions can disagree and only one is documented as the source.",
    "duplicated_test_block": "The assertions already exist in another test, so one contract now has to be remembered in two places.",
    "broad_except": "The clause catches a type wider than anything the body it guards can raise.",
    "swallowed_exception": "The handler catches and then neither logs nor re-raises, so the failure leaves no trace behind.",
    "unclosed_resource": "A file, socket, cursor or lock is opened on a path that can leave without closing it -- no `with`, no `finally`.",
    "redundant_work": "The same value is computed twice in one flow where the first result is still in scope.",
    "work_in_loop": "A call that goes to the database, the network or the filesystem is made once per iteration of a loop over a set.",
    "removed_dependency": "A package is gone from the runtime requirements while an import of it remains.",
    "missing_assertion": "The test runs the code and asserts nothing about the result.",
    "hardcoded_endpoint": "The test writes a path literally instead of resolving it the way the application does.",
    "null_deref": "The result of a call that can be absent is used -- an attribute, an index, a call -- with no check that it is there.",
    "off_by_one": "The index, slice or offset is one away from the range the operation needs: a boundary included that should be left out, or left out that should be in.",
    "unguarded_dict_access": "A key that came from outside indexes a mapping directly, so a key that is not there raises instead of being handled.",
    "missing_lock": "A read-modify-write of state that more than one caller can reach runs with nothing holding the value still between the read and the write.",
    "secret_in_log": "A value obtained as a secret is written to a log in clear.",
    "ssrf_unvalidated_fetch": "An address the caller controls is fetched with no restriction on where it may point.",
    "unsafe_deserialization": "Untrusted data reaches a decoder that can construct objects or run code -- `pickle`, `yaml.load`, `marshal`.",
    "divergent_change": "One module is changed for reasons that have nothing to do with each other.",
    "path_traversal": "A path is built from a value the caller controls and opened, with nothing keeping it inside the directory it is meant to stay in.",
    "open_redirect": "A redirect is sent to a location the request supplied, without the restriction applied where other redirect targets are chosen.",
    "mass_assignment": "A request body is bound wholesale onto a stored object, so fields the explicit list for that object leaves out can be written from outside.",
    "unbounded_resource": "A response, file or query result is read whole into memory with no limit on how large it may be.",
    "float_money": "A monetary amount is held or computed in a binary floating type, so the rounding it introduces reaches a stored or charged value.",
    "naive_datetime": "A timestamp is created or compared without a timezone, so what it means depends on where the code runs.",
    "blocking_call_in_async": "A synchronous call that waits -- a socket, a file, a sleep -- runs inside a coroutine, where an awaitable form of the same call exists.",
    "mutable_default_argument": "A parameter defaults to a mutable object, so the value outlives the call and is shared by every later one.",
    "regex_denial_of_service": "A pattern with nested or overlapping repetition is matched against input from outside, where the time it takes grows faster than the input does.",
    "breaking_public_api": "A name callers outside this change depend on -- an exported symbol, a flag, an attribute a subclass overrides -- is removed, renamed or narrowed.",
    "missing_migration": "A stored model's shape is changed with no accompanying migration.",
    "wrong_assertion_target": "A test asserts on something other than the effect it names -- the input it just built, a mock's own return value, or a field the code under test never writes -- so it passes whatever that code does.",
    "leaky_test_state": "A test leaves process state behind, or depends on state another test left, so its result depends on which tests ran before it.",
    "unsafe_default": "A default value turns a protection off or removes a bound: verification skipped, a timeout unlimited, a debug path left on.",
    "contradictory_setting": "Two settings that are read together state incompatible things, so one of them cannot take effect.",
    "removed_config_key": "A configuration key or target -- a host, an endpoint, a queue, a credential name -- is deleted while code that reads it stays, so the reader silently falls back to a value nobody chose.",
    "missing_timeout": "A network call, a subprocess or a lock acquisition is made with no timeout, so a peer that never answers holds the caller for ever.",
    "missing_transaction": "Two or more writes that only make sense together are made without a transaction, so a failure between them leaves the store half-written.",
    "retry_without_bound": "A retry has no attempt limit or no wait between attempts, so a dependency that is failing is called again as fast as it can refuse.",
    "dynamic_code_execution": "A value that came from outside reaches `eval`, `exec`, `pickle.loads` of an expression, or an import that runs what it names.",
    "insecure_transport": "A request is made with certificate verification disabled, or over plain HTTP, where the data it carries is not public.",
    "unawaited_coroutine": "A coroutine is called without being awaited or scheduled, so the work it describes never runs and its failure is never seen.",
    "disabled_test": "A test is skipped unconditionally, or returns before its assertions, so it reports success without exercising anything.",
    "global_state_mutation": "Module-level mutable state is written while handling a request or a task, so one caller sees what another left behind.",
    "lossy_conversion": "A value is converted to a narrower type -- an integer from a fraction, a float from an exact decimal, a truncation -- where the part that is lost changes the result.",
    "encoding_assumption": "Bytes are decoded, or text encoded, under an assumed encoding that the source does not promise.",
    "unsafe_temp_file": "A temporary file or directory is created at a predictable path, or with default permissions, where another process can reach it.",
    "ignored_return_value": "A call whose return value reports whether the work succeeded is made as a statement, so the failure it reports is dropped.",
    "stale_cache_write": "A write updates the store without invalidating or updating the cache that is read for the same value.",
    "overly_permissive_permission": "A file mode, bucket policy or object ACL is set wider than the code needs -- world-writable, publicly readable -- where comparable resources are narrower.",
    "insecure_randomness": "A token, key, password or identifier that has to be unguessable is drawn from a non-cryptographic random source.",
    "loop_without_progress": "A loop's exit condition is not advanced on every path through its body, so an input exists for which it never ends.",
    "check_then_act_race": "A value is read, a decision is made from it, and the decision is acted on as though the value had not changed in between -- a balance checked then debited, stock checked then reserved, a row checked then inserted.",
    "missing_duplicate_guard": "An operation that can arrive twice -- a webhook, a retried request, a queue message -- is carried out with nothing that recognises the repeat, so the second delivery does the work again.",
}


def _check(dataset: str, version: str | None) -> None:
    if dataset not in FORMATS:
        raise KeyError(f"unknown dataset {dataset!r}; have {', '.join(DATASETS_KNOWN)}")
    if version not in (None, PROMPT_VERSION):
        raise KeyError(f"prompt version {version!r} is retired; the only version is {PROMPT_VERSION!r}")


def types(dataset: str, version: str | None = None) -> list[str]:
    """The names the answer schema constrains a finding's type to."""
    _check(dataset, version)
    return list(CATALOGUE)


def definitions(dataset: str, version: str | None = None) -> dict[str, str]:
    """Each name's definition, as the verifier is shown it."""
    _check(dataset, version)
    return dict(CATALOGUE)


def system(dataset: str, version: str = PROMPT_VERSION) -> str:
    """The full system prompt: the instructions with the catalogue before `How to look`."""
    _check(dataset, version)
    head, tail = INSTRUCTIONS.split("## How to look")
    kinds = "\n".join(f"- `{name}` -- {text}" for name, text in CATALOGUE.items())
    return (
        f"{head.format(format=FORMATS[dataset])}"
        f"## The kinds of defect you report\n\nReport only these, and nothing else:\n\n"
        f"{kinds}\n\n"
        f"## How to look{tail}"
    )
