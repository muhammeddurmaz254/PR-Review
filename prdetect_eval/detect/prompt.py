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

# The default is the best measured version. On halka_bench, where the project now
# measures, that is v6: F1 0.808 against v5's 0.804, forty-three of forty-nine
# located, and the widest definition of a defect any version has carried.
PROMPT_VERSION = "review/v6"

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

FORMATS = {"demo_repo": WHOLE_FILE_FORMAT, "swrbench": HUNK_FORMAT,
           "halka": WHOLE_FILE_FORMAT, "zincir": WHOLE_FILE_FORMAT}


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


# halka_bench names forty-one kinds, thirty-three of them in primary scope. The
# definitions are the corpus's own, translated: it grounds each one in a
# convention its base branch establishes in more than one place, and v3 measured
# what happens when a catalogue is written from what the words suggest instead.
#
# Its repository is small enough to carry whole -- fifty-nine files, about twelve
# thousand tokens -- and two thirds of its defects are only legible against code
# the pull request does not touch. That is what `--with-repo` is for.

HALKA_TYPES = {
    "missing_authz_check": "A mutation or object access is guarded by a weaker permission "
                           "predicate than its siblings use for the same kind of operation.",
    "crossfile_ownership": "A query does not scope to the tenant; the scoping contract is "
                           "shared between the caller and the query and one side dropped it.",
    "crossfile_data_exposure": "Serialization bypasses the closed field list and sends "
                               "sensitive columns to the client.",
    "crossfile_error_propagation": "A failure the callee reports is swallowed by the caller "
                                   "and reported as success.",
    "crossfile_idempotency": "The idempotency key changes on every run, so the duplicate "
                             "guard never engages.",
    "crossfile_ordering": "An ordering guarantee the consumer depends on was removed on the "
                          "producer side.",
    "crossfile_unit_mismatch": "A value is passed in a different unit from the one the callee "
                               "expects.",
    "sql_injection": "User data is concatenated into SQL text.",
    "command_injection": "User data is concatenated into shell command text.",
    "xss": "User data is written unescaped into an HTML response.",
    "error_detail_disclosure": "Exception text or a stack trace is put into the client response.",
    "hardcoded_credential": "A secret value is written into the source.",
    "weak_crypto_primitive": "A broken digest, predictable randomness, or a secret compared in "
                             "non-constant time.",
    "unvalidated_passthrough": "Request data is forwarded as-is into a call that expects "
                               "validated input.",
    "wrong_argument": "Arguments are passed to a call in the wrong order or the wrong role.",
    "wrong_data_source": "A value is read from a cache the repository documents as able to go "
                         "stale; the authority is elsewhere.",
    "wrong_state_check": "A state comparison is made against a value the state machine does "
                         "not contain.",
    "silent_overwrite": "A write clobbers existing fields instead of merging them.",
    "unreachable_code": "Statements control flow can never reach.",
    "unused_symbol": "A symbol is defined and referenced from nowhere.",
    "misleading_name": "A name describes the behaviour wrongly by the repository's own naming "
                       "convention.",
    "duplicated_block": "A copy of a block that already exists in the repository.",
    "duplicated_config": "A second copy of a configuration value documented as single-source.",
    "duplicated_test_block": "A copy of an existing test block.",
    "broad_except": "An exception type far wider than the failure being handled is caught.",
    "swallowed_exception": "An exception is caught and neither logged nor reported to the caller.",
    "unclosed_resource": "An opened resource is not closed by a `with` block or a `close` call.",
    "redundant_work": "The same expensive computation is done more than once in one flow.",
    "work_in_loop": "A query with a batch alternative is run one row at a time inside a loop.",
    "removed_dependency": "A package imported at runtime was dropped from the runtime "
                          "dependency list.",
    "removed_network_config": "A network target the code still uses was removed from "
                              "configuration.",
    "missing_assertion": "A test asserts nothing.",
    "hardcoded_endpoint": "A test uses a literal path string instead of the URL map.",
    "null_deref": "The result of a call that can return None is used unchecked.",
    "off_by_one": "A pagination offset breaks the repository's one-based page contract.",
    "unguarded_dict_access": "Unguarded dictionary access with a key that came from outside.",
    "missing_lock": "A read-modify-write runs without the repository's locking contract.",
    "secret_in_log": "A raw secret is written to the log stream in clear text.",
    "ssrf_unvalidated_fetch": "A user-controlled address is fetched, bypassing the allow-list gate.",
    "unsafe_deserialization": "Untrusted data is handed to a loader that can execute code.",
    "divergent_change": "A module is changed for unrelated reasons.",
}


# --- Section 0.5: the catalogue is the product's, not the dataset's ----------
#
# ``TAXONOMIES_BROAD`` used to read ``{**TAXONOMIES_V3, "halka": ...}``: it
# widened the catalogue for one corpus. Every other dataset fell through to v6's list,
# byte for byte, so "review/v6-broad on demo_repo" was review/v6 under another
# name and the question the experiment exists to answer -- does a catalogue that
# helps on one repository still help on another -- was never asked.
#
# The catalogue is a property of the product. ``catalog/product_catalog.json``
# holds it, both corpora draw their type names from it, and these five are the
# names zincir_bench needed that halka_bench had no case for.
SHARED_ADDITIONS = {
    "wrong_assertion_target": "A test asserts on something other than the effect it names -- "
                              "the input it just built, a mock's own return value, or a field "
                              "the code under test never writes -- so it passes whatever that "
                              "code does.",
    "leaky_test_state": "A test leaves process state behind, or depends on state another test "
                        "left, so its result depends on which tests ran before it.",
    "unsafe_default": "A default value turns a protection off or removes a bound -- "
                      "verification skipped, a timeout unlimited, a debug path left on -- "
                      "where the same setting is stated explicitly elsewhere.",
    "contradictory_setting": "Two settings that are read together state incompatible things, "
                             "so one of them cannot take effect.",
    "removed_config_key": "A configuration key is deleted while code that reads it stays, so "
                          "the reader silently falls back to a value nobody chose.",
}

# The whole product vocabulary, one definition per name. halka keeps its own
# forty-one entry list so the two dozen measurements already taken against it
# stay byte-identical; ``review/v6-shared`` below is where the two corpora are
# handed the *same* catalogue, which is the only way the portability question
# has an answer rather than an assumption.
SHARED_TYPES = {**HALKA_TYPES, **SHARED_ADDITIONS}

TAXONOMIES = {"demo_repo": DEMO_REPO_TYPES, "swrbench": SWRBENCH_TYPES,
              "halka": HALKA_TYPES, "zincir": SHARED_TYPES}
TAXONOMIES_V3 = {"demo_repo": DEMO_REPO_TYPES, "swrbench": SWRBENCH_TYPES_V3,
                 "halka": HALKA_TYPES, "zincir": SHARED_TYPES}

# --- Deney A: temellendirme mi, genişlik mi? -------------------------------
#
# The same forty-one type names, defined the way a universal catalogue would
# define them: from the class of defect, not from how this repository expresses
# it. Written without consulting halka_bench. Where the grounded definition
# names something local -- "weaker than its siblings use", "the repository
# documents", "the repository's one-based page contract" -- the generic one
# states the class instead.
#
# The question this answers: v3 doubled the primary metric by writing
# definitions from the corpus's own examples. If generic wording scores the
# same, that gain was about writing them *carefully*, and a shared catalogue
# carries to any repository. If it scores worse, grounding is the thing that
# worked and every deployment has to pay for its own.

HALKA_TYPES_GENERIC = {
    "missing_authz_check": "An endpoint or object access is not authorized, or the check it "
                           "does is weaker than the operation it guards requires.",
    "crossfile_ownership": "A query is not restricted to the tenant, account or owner whose "
                           "data it is supposed to return.",
    "crossfile_data_exposure": "Serialization sends fields to the client that were meant to "
                               "stay internal.",
    "crossfile_error_propagation": "A failure reported by a callee is swallowed by its caller "
                                   "and reported as success.",
    "crossfile_idempotency": "A retried or repeated operation is processed more than once "
                             "because its duplicate guard never engages.",
    "crossfile_ordering": "An ordering guarantee a consumer relies on is not provided by the "
                          "producer.",
    "crossfile_unit_mismatch": "A value crosses a boundary in a different unit or scale from "
                               "the one the other side expects.",
    "sql_injection": "Untrusted input reaches the text of an SQL statement instead of being "
                     "bound as a parameter.",
    "command_injection": "Untrusted input reaches the text of a shell command.",
    "xss": "Untrusted input is written into an HTML response without escaping.",
    "error_detail_disclosure": "An exception message or stack trace is returned to the client.",
    "hardcoded_credential": "A password, key or token is written literally into the source.",
    "weak_crypto_primitive": "A broken or unsuitable cryptographic choice: a weak digest, "
                             "predictable randomness, or a secret compared in variable time.",
    "unvalidated_passthrough": "Request data is forwarded unchecked into a call that expects "
                               "validated input.",
    "wrong_argument": "A call receives arguments in the wrong order, or the wrong value in a "
                      "position.",
    "wrong_data_source": "A value is read from a source that can be stale or wrong when an "
                         "authoritative one exists.",
    "wrong_state_check": "A state is compared against a value the state machine cannot hold.",
    "silent_overwrite": "A write replaces existing data that should have been merged or "
                        "preserved.",
    "unreachable_code": "Statements that control flow can never reach.",
    "unused_symbol": "A symbol is defined and never referenced.",
    "misleading_name": "A name describes behaviour the code does not have, so callers read it "
                       "wrongly.",
    "duplicated_block": "A block of logic is copied where the existing one could be reused.",
    "duplicated_config": "A configuration value is written in a second place, so the two can "
                         "drift apart.",
    "duplicated_test_block": "A test block is copied rather than shared, so a change has to be "
                             "remembered twice.",
    "broad_except": "An exception clause catches far more than the failure being handled.",
    "swallowed_exception": "An exception is caught and neither logged nor re-raised.",
    "unclosed_resource": "A file, socket or handle is opened and not reliably closed.",
    "redundant_work": "The same expensive computation is performed more than once where one "
                      "result would do.",
    "work_in_loop": "A query or call is issued per iteration where a batch form exists.",
    "removed_dependency": "A package the code imports at runtime is no longer among its runtime "
                          "dependencies.",
    "removed_network_config": "A host, endpoint or credential the code still uses is no longer "
                              "present in configuration.",
    "missing_assertion": "A test exercises code without asserting anything about the result.",
    "hardcoded_endpoint": "A test hardcodes a URL or path instead of resolving it the way the "
                          "application does.",
    "null_deref": "A value that can be absent is used without checking for it.",
    "off_by_one": "An index, bound or offset is out by one.",
    "unguarded_dict_access": "A key from outside the program is used to index a mapping without "
                             "handling its absence.",
    "missing_lock": "A read-modify-write runs without the mutual exclusion its invariant needs.",
    "secret_in_log": "A credential or token is written to a log.",
    "ssrf_unvalidated_fetch": "A request is made to an address the caller controls, without "
                              "restricting where it may point.",
    "unsafe_deserialization": "Untrusted data is handed to a decoder that can construct or "
                              "execute arbitrary objects.",
    "divergent_change": "One module is repeatedly changed for unrelated reasons.",
}

# --- Deney B: katalog uzunluğu zarar veriyor mu? ---------------------------
#
# Every type of all three corpora at once: fifty-four names, of which thirteen
# cannot occur in halka_bench at all. A shared catalogue would look like this
# from any single repository's point of view -- mostly irrelevant. Measured
# against the forty-one, the difference is what breadth costs by itself,
# separately from how the definitions are worded.

HALKA_TYPES_WIDE = {**HALKA_TYPES, **DEMO_REPO_TYPES, **SWRBENCH_TYPES_V3}

TAXONOMIES_GENERIC = {**TAXONOMIES_V3, "halka": HALKA_TYPES_GENERIC,
                      "zincir": {**HALKA_TYPES_GENERIC, **SHARED_ADDITIONS}}

# --- Deney C: repo-agnostik kelimeler, kanıt talep eden şekil ---------------
#
# Deney A wrote the same forty-one types the way a catalogue describes a class,
# and tripled the false alarms while leaving localization and naming untouched.
# Reading the two catalogues against each other suggested why: the definitions
# that won name **a comparison the reviewer has to go and find** -- "weaker than
# its siblings use", "documented as able to go stale", "the one-based page
# contract" -- and none of them names this repository. The generic ones assert a
# property, which a model can claim from the accused line alone.
#
# So this catalogue keeps the evidence and drops the locality: every definition
# says what would establish it, in words that belong to no particular codebase.
# If it lands near the grounded numbers, a shared catalogue is possible and the
# per-deployment cost is writing tests rather than descriptions. If it lands
# near the generic ones, demanding evidence is not enough and grounding really
# does require knowing the repository.

HALKA_TYPES_EVIDENCE = {
    "missing_authz_check": "The check guarding this operation is weaker than the one its "
                           "sibling operations of the same kind apply.",
    "crossfile_ownership": "A query does not apply the tenant or owner scoping that the "
                           "sibling queries around it apply.",
    "crossfile_data_exposure": "Serialization emits fields that the field list used by the "
                               "sibling serializers excludes.",
    "crossfile_error_propagation": "The callee reports a failure and the caller reports "
                                   "success; the two disagree about what happened.",
    "crossfile_idempotency": "The key the duplicate guard is keyed on is not stable across "
                             "runs, so the guard it feeds can never match.",
    "crossfile_ordering": "The consumer depends on an order the producer no longer "
                          "establishes.",
    "crossfile_unit_mismatch": "The value passed is in a different unit from the one the "
                               "callee's name, signature or docstring declares.",
    "sql_injection": "A runtime value is placed in the query text rather than in the "
                     "parameters the driver binds.",
    "command_injection": "A runtime value is placed in the command text rather than in an "
                         "element of an argument list.",
    "xss": "A value reaches the response unescaped where the same value is escaped on the "
           "other paths that render it.",
    "error_detail_disclosure": "The response carries exception text or a trace where sibling "
                               "handlers return a code alone.",
    "hardcoded_credential": "A secret is a literal here where its neighbours read the same "
                            "kind of value from configuration.",
    "weak_crypto_primitive": "The primitive chosen is weaker than the one used elsewhere for "
                             "the same purpose: a broken digest, ordinary randomness for a "
                             "secret, or a comparison that returns early.",
    "unvalidated_passthrough": "Request data reaches a call that its other call sites reach "
                               "only after validating.",
    "wrong_argument": "The arguments do not match the order or the roles the callee's "
                      "signature declares.",
    "wrong_data_source": "The value is read from a source that can lag, while the "
                         "authoritative one is used for the same value elsewhere.",
    "wrong_state_check": "A state is compared against a value nothing in the code ever "
                         "assigns to it.",
    "silent_overwrite": "A write replaces a structure whose other writers merge into it.",
    "unreachable_code": "The control flow above these statements always leaves before them.",
    "unused_symbol": "The symbol is defined and nothing that can see it refers to it.",
    "misleading_name": "The name says something the body does not do, judged against the "
                       "naming convention its neighbours follow.",
    "duplicated_block": "The same logic already exists elsewhere and this is a second copy "
                        "rather than a use of it.",
    "duplicated_config": "The value is defined a second time, so the two definitions can "
                         "disagree and only one is documented as the source.",
    "duplicated_test_block": "The assertions already exist in another test, so one contract "
                             "now has to be remembered in two places.",
    "broad_except": "The clause catches a type wider than anything the body it guards can "
                    "raise.",
    "swallowed_exception": "The handler neither logs nor re-raises, where sibling handlers "
                           "do one or the other.",
    "unclosed_resource": "The resource is opened without the closing form its sibling opens "
                         "use.",
    "redundant_work": "The same value is computed twice in one flow where the first result "
                      "is still in scope.",
    "work_in_loop": "A call is made per iteration although a form that takes the whole set "
                    "exists and is used elsewhere.",
    "removed_dependency": "A package is gone from the runtime requirements while an import "
                          "of it remains.",
    "removed_network_config": "A target is gone from configuration while a use of it remains.",
    "missing_assertion": "The test runs the code and asserts nothing about the result, where "
                         "its sibling tests assert.",
    "hardcoded_endpoint": "The test writes a path literally instead of resolving it the way "
                          "the application does.",
    "null_deref": "The result of a call that can be absent is used without the check its "
                  "other call sites make.",
    "off_by_one": "The index or offset is one away from the convention its other call sites "
                  "follow.",
    "unguarded_dict_access": "A key from outside indexes a mapping without the guard sibling "
                             "lookups use.",
    "missing_lock": "A read-modify-write runs without the mutual exclusion that sibling "
                    "mutations of the same state take.",
    "secret_in_log": "A value obtained as a secret is written to a log in clear.",
    "ssrf_unvalidated_fetch": "An address the caller controls is fetched without the "
                              "restriction applied at the other fetch sites.",
    "unsafe_deserialization": "Untrusted data reaches a decoder that can construct objects, "
                              "where a safe decoder is used for the same job elsewhere.",
    "divergent_change": "One module is changed for reasons that have nothing to do with each "
                        "other.",
}

TAXONOMIES_EVIDENCE = {**TAXONOMIES_V3, "halka": HALKA_TYPES_EVIDENCE,
                      "zincir": {**HALKA_TYPES_EVIDENCE, **SHARED_ADDITIONS}}

# --- Deney D: referansı yapısal olarak adlandır -----------------------------
#
# Deney C recovered every true positive and every naming decision with wording
# that names no repository, and stopped one gap short: nine false alarms against
# the grounded catalogue's four, all six of the extra ones on clean cases and all
# of them under the comparative types.
#
# Reading them gave the reason. C says "weaker than its siblings use" without
# saying which siblings, so on a clean pull request the model finds *some*
# neighbour that differs and calls it a violation. The grounded catalogue was
# tighter because it named a reference that can be looked up and therefore
# failed to be found: "documented as able to go stale", "the one-based page
# contract".
#
# So D keeps C's demand for a comparison and adds the missing half: the
# comparison must be against something the model was actually shown, named by
# where it is. Not "its siblings" -- "the other endpoints in this same file".
# That is still repository-agnostic; it is a location, not a convention.

HALKA_TYPES_LOCATED = {
    "missing_authz_check": "The check guarding this operation is weaker than the check the "
                           "other operations of the same kind in this same file apply.",
    "crossfile_ownership": "A query omits the tenant or owner scoping that the other queries "
                           "in the same module apply to the same table.",
    "crossfile_data_exposure": "Serialization emits a field that the explicit field list "
                               "shown for this same object leaves out.",
    "crossfile_error_propagation": "A callee shown in this change reports failure, and its "
                                   "caller shown here returns success anyway.",
    "crossfile_idempotency": "The value the duplicate guard is keyed on is computed fresh at "
                             "the call, so two runs of the same work produce two keys.",
    "crossfile_ordering": "A consumer shown in this change reads in an order that the "
                          "producer shown here does not establish.",
    "crossfile_unit_mismatch": "The value passed is in a different unit from the one the "
                               "callee's parameter name, signature or docstring declares.",
    "sql_injection": "A runtime value is placed in the query text rather than in the "
                     "parameter sequence passed alongside it.",
    "command_injection": "A runtime value is placed in the command text rather than in an "
                         "element of the argument list passed alongside it.",
    "xss": "A value reaches the response unescaped on one path where another path shown for "
           "the same value escapes it.",
    "error_detail_disclosure": "The response body carries exception text or a trace, where "
                               "another handler in this same file returns a status alone.",
    "hardcoded_credential": "A secret appears as a literal, where another value of the same "
                            "kind in this same file is read from configuration.",
    "weak_crypto_primitive": "The primitive used here is weaker than the one used for the "
                             "same purpose at another site in this change: a broken digest, "
                             "ordinary randomness for a secret, or a comparison that can "
                             "return early.",
    "unvalidated_passthrough": "Request data reaches a call that another call site of the "
                               "same function, shown here, reaches only after validating.",
    "wrong_argument": "The arguments do not match the order or the roles the callee's "
                      "signature, shown in this change, declares.",
    "wrong_data_source": "The value is read from a field this change also writes as a cached "
                         "copy, while the computation it caches is available at this point.",
    "wrong_state_check": "A state is compared against a value that no assignment in this "
                         "change ever gives it.",
    "silent_overwrite": "A write replaces a structure that another write to the same "
                        "structure, shown here, merges into.",
    "unreachable_code": "Every path through the lines above these ones returns or raises "
                        "before reaching them.",
    "unused_symbol": "The symbol is defined here and no line shown in this change refers "
                     "to it.",
    "misleading_name": "The name follows a pattern that other names in this same file use "
                       "for different behaviour from this body's.",
    "duplicated_block": "This block repeats, statement for statement, a block that already "
                        "exists at another place shown in this change.",
    "duplicated_config": "This defines a value that another line shown in this change "
                         "already defines, and only one of the two can be the source.",
    "duplicated_test_block": "These assertions repeat, statement for statement, assertions "
                             "that another test in this same file already makes.",
    "broad_except": "The clause names an exception type wider than any the statements it "
                    "guards can raise.",
    "swallowed_exception": "The handler neither logs nor re-raises, where another handler in "
                           "this same file does one or the other.",
    "unclosed_resource": "The resource is opened without the `with` or `close` that another "
                         "open of the same kind in this same file uses.",
    "redundant_work": "The same call with the same arguments is made twice in one flow, with "
                      "the first result still in scope at the second.",
    "work_in_loop": "A call is made once per item although a form taking the whole collection "
                    "is defined or used at another site in this change.",
    "removed_dependency": "A package is removed from the runtime requirements shown here "
                          "while an import of it remains in the code shown here.",
    "removed_network_config": "A target is removed from the configuration shown here while a "
                              "use of it remains in the code shown here.",
    "missing_assertion": "The test calls the code and makes no assertion, where another test "
                         "in this same file asserts on its result.",
    "hardcoded_endpoint": "The test writes a path as a literal where the application code "
                          "shown in this change resolves the same path through a map.",
    "null_deref": "The result of a call is used without the check that another call site of "
                  "the same function, shown here, makes on it.",
    "off_by_one": "The index or offset differs by one from the convention another call site "
                  "of the same function, shown here, follows.",
    "unguarded_dict_access": "A key from outside indexes a mapping without the guard another "
                             "lookup in this same file uses.",
    "missing_lock": "A read-modify-write runs without the lock that another mutation of the "
                    "same state, shown in this change, takes.",
    "secret_in_log": "A value this change obtains as a secret is passed to a logging call in "
                     "clear.",
    "ssrf_unvalidated_fetch": "An address the caller controls is fetched without the "
                              "restriction that another fetch shown in this change applies.",
    "unsafe_deserialization": "Untrusted data reaches a decoder that can construct objects, "
                              "where another decode shown in this change uses a safe one.",
    "divergent_change": "This change edits one module for two reasons that share nothing.",
}

TAXONOMIES_LOCATED = {**TAXONOMIES_V3, "halka": HALKA_TYPES_LOCATED,
                      "zincir": {**HALKA_TYPES_LOCATED, **SHARED_ADDITIONS}}

# --- Deney E: yer adlandırılmış ama diff'e kilitli değil --------------------
#
# D named the reference and clawed back two false alarms, and lost three true
# positives doing it. All three name a reference that lives in code the pull
# request does not touch -- the original of a duplicated block, the batch
# selector a loop should have used -- and D had written "shown in this change",
# which excludes them by construction. Thirty-three of this corpus's forty-nine
# defects are grounded that way, so the restriction was aimed at two thirds of
# the answer key.
#
# E is D with exactly that phrase lifted and nothing else changed, so the
# difference between the two runs is the diff restriction alone. The reference
# is still a place -- "another place in this codebase", "the callee's signature"
# -- and still names no repository.

_UNRESTRICT = (
    (", shown in this change,", ""),
    (", shown here,", ""),
    (" shown in this change", " in this codebase"),
    (" shown here", " in this codebase"),
    (" shown for this same object", " for this same object"),
    ("at another site in this change", "at another site in this codebase"),
    ("another test in this same file", "another test in this codebase"),
)


def _unrestrict(text: str) -> str:
    for before, after in _UNRESTRICT:
        text = text.replace(before, after)
    return text


HALKA_TYPES_REACHABLE = {name: _unrestrict(text)
                         for name, text in HALKA_TYPES_LOCATED.items()}

TAXONOMIES_REACHABLE = {**TAXONOMIES_V3, "halka": HALKA_TYPES_REACHABLE,
                      "zincir": {**HALKA_TYPES_REACHABLE, **SHARED_ADDITIONS}}

# --- Deney F: C disiplini ikinci bir korpusta -------------------------------
#
# C beat the other agnostic variants on halka_bench, and the obvious objection is
# that its author knows halka_bench. The wording can be checked -- no definition
# names the repository, and a test asserts it -- but the *choices* cannot: "the
# check its other call sites make" may have been picked because that is how this
# corpus happens to work.
#
# demo_repo is the control. Its eight definitions have not been touched since
# review/v1, so they are a grounded baseline nobody has been tuning, and its
# cases were never read while writing what follows. If the discipline carries,
# the gap here should look like the gap there.

DEMO_REPO_TYPES_EVIDENCE = {
    "authz": "The check guarding this operation is weaker than the check the other "
             "operations of the same kind apply, or there is none where they have one.",
    "business_logic": "A value is computed, converted or reset in a way that contradicts the "
                      "rule the surrounding code states for it -- a unit, a default, a bound "
                      "another line declares.",
    "data_exposure": "Serialization emits a field that the explicit list of fields for this "
                     "same object leaves out.",
    "error_handling": "A callee reports a failure and its caller reports success, or the "
                      "handler neither logs nor re-raises where another handler does one of "
                      "the two.",
    "idempotency": "The value the duplicate guard is keyed on is computed fresh at the call, "
                   "so two runs of the same work produce two keys and the guard never "
                   "matches.",
    "injection": "A runtime value is placed in the text of a query or command rather than in "
                 "the parameter or argument sequence passed alongside it.",
    "race_condition": "A read-modify-write runs without the mutual exclusion that another "
                      "mutation of the same state takes.",
    "secrets": "A value obtained as a credential is written where it can be read back -- a "
               "log call, a response, a literal in the source -- while the same kind of value "
               "is read from configuration on another line.",
}

TAXONOMIES_DEMO_EVIDENCE = {**TAXONOMIES_V3, "demo_repo": DEMO_REPO_TYPES_EVIDENCE}

# --- Deney G: geniş katalog + jenerik tanımlar ------------------------------
#
# B put all fifty-four names in the prompt and lost 0.242 of F1, eleven of its
# eighteen false alarms arriving under kinds the corpus does not contain. But B
# carried halka_bench's *grounded* definitions for its own forty-one and each
# other corpus's for theirs, so it confounded two things: how many names there
# are, and whether their definitions were written with a repository in hand.
#
# This is the configuration a real deployment would actually ship -- every name
# anyone might need, none of the definitions written against the code being
# reviewed. If it lands near C the breadth is affordable and B measured the
# definitions; if it lands near B the breadth is the cost and a shared catalogue
# has to be narrowed per pull request after all.

SWRBENCH_TYPES_EVIDENCE = {
    "F.1 Interface": "A call is handed an object or a value of a different kind from the one "
                     "the callee's signature declares, or a public name callers rely on is "
                     "removed or narrowed.",
    "F.2 Logic": "The expression is wrong for input it will see: a condition true in a case "
                 "it should not be, a value built from the wrong parts, a format string that "
                 "never interpolates.",
    "F.3 Resource": "A name, path or setting is not available in the form it is used: a symbol "
                    "referenced but never imported, a parameter computed and never passed on, "
                    "an entry point that no longer works the documented way.",
    "F.4 Check": "The guarding is wrong: missing where an invalid value flows on, applied on "
                 "one path and not the one beside it, asserting something that can be false, "
                 "or catching so widely that unrelated failures vanish.",
    "F.5 Support": "The code assumes a language version, a library version or a dependency "
                   "that will not always be there.",
}

HALKA_TYPES_WIDE_EVIDENCE = {**HALKA_TYPES_EVIDENCE, **DEMO_REPO_TYPES_EVIDENCE,
                             **SWRBENCH_TYPES_EVIDENCE}

TAXONOMIES_WIDE_EVIDENCE = {**TAXONOMIES_V3, "halka": HALKA_TYPES_WIDE_EVIDENCE,
                            "zincir": {**HALKA_TYPES_WIDE_EVIDENCE, **SHARED_ADDITIONS}}

# --- Deney H: tekilleştirilmiş ve genişletilmiş katalog ---------------------
#
# G merged three taxonomies and lost a quarter of F1. The extra false alarms did
# not arrive under the kinds that could not occur -- `F.1 Interface` produced one
# -- but under the coarse synonyms the merge introduced: `authz` beside
# `missing_authz_check`, `injection` beside `sql_injection`, `error_handling`
# beside three separate halka names. The model reported under the coarse name and
# the label carried the precise one.
#
# So this catalogue is the same test without that fault. Nothing is a synonym of
# anything else, and twelve classes are added that this corpus cannot contain at
# all -- a deployment ships names for defects it may never see. If it lands near
# C, breadth is affordable once the catalogue is a catalogue; if it lands near G,
# the count itself is the cost and a shared one has to be narrowed per pull
# request.

BEYOND_CORPUS = {
    "path_traversal": "A path is built from a value the caller controls and used to open a "
                      "file, without the containment the other file opens in this codebase "
                      "apply.",
    "open_redirect": "A redirect is sent to a location the request supplied, without the "
                     "restriction applied where other redirect targets are chosen.",
    "mass_assignment": "A request body is bound wholesale onto a stored object, so fields the "
                       "explicit list for that object leaves out can be written from outside.",
    "insecure_default": "A default turns a protection off -- verification skipped, debug left "
                        "on, a permissive origin -- where the same setting is explicit "
                        "elsewhere.",
    "unbounded_resource": "A response, file or query result is read whole into memory with no "
                          "limit, where a bound or a stream is used for the same kind of "
                          "input elsewhere.",
    "float_money": "A monetary amount is held or computed in a floating type, where the "
                   "surrounding code keeps money in an exact one.",
    "naive_datetime": "A timestamp is created or compared without a timezone, where the other "
                      "timestamps in this codebase carry one.",
    "blocking_call_in_async": "A synchronous call that waits -- a socket, a file, a sleep -- "
                              "runs inside a coroutine, where an awaitable form of the same "
                              "call exists.",
    "mutable_default_argument": "A parameter defaults to a mutable object, so the value "
                                "outlives the call and is shared by every later one.",
    "regex_denial_of_service": "A pattern with nested or overlapping repetition is matched "
                               "against input from outside, where the time it takes grows "
                               "faster than the input does.",
    "breaking_public_api": "A name callers outside this change depend on -- an exported "
                           "symbol, a flag, an attribute a subclass overrides -- is removed, "
                           "renamed or narrowed.",
    "missing_migration": "A stored model's shape is changed with no accompanying migration, "
                         "where every other change to that model has one.",
}

# One name per class: the halka forty-one, none of which is a synonym of another,
# plus twelve classes it cannot contain.
HALKA_TYPES_BROAD = {**HALKA_TYPES_EVIDENCE, **BEYOND_CORPUS}

TAXONOMIES_BROAD = {**TAXONOMIES_V3, "halka": HALKA_TYPES_BROAD,
                    "zincir": {**HALKA_TYPES_BROAD, **SHARED_ADDITIONS}}






TAXONOMIES_WIDE = {**TAXONOMIES_V3, "halka": HALKA_TYPES_WIDE,
                      "zincir": {**HALKA_TYPES_WIDE, **SHARED_ADDITIONS}}


# --- The portability run ----------------------------------------------------
#
# The same catalogue, the same words, on both corpora: forty-six names, of which
# halka has positives for thirty-nine and zincir for eight. Everything else in
# the two runs differs -- domain, architecture, conventions, file layout -- so a
# gap between them is a gap in the catalogue's reach and not in its wording.
#
# The names each corpus has no case for are not slack. Section 0.4 makes a
# report under one of them a false alarm, which is what a deployed analyser's
# extra names actually cost.
TAXONOMIES_SHARED = {**TAXONOMIES_V3, "halka": SHARED_TYPES, "zincir": SHARED_TYPES}


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

# v4 adds one field: the line the finding is about, copied from the prompt. It
# is not for the reader -- it is the handle stage [5] needs. A claim whose quote
# appears nowhere in what was shown is describing code that was not there, and
# that is decidable by string comparison rather than by asking the model again,
# which was measured to reproduce the original mistake.

INSTRUCTIONS_V4 = INSTRUCTIONS_V3.replace(
    '{"findings": [{"file": "...", "line": 0, "type": "...", "title": "...", "confidence": 0.0}]}',
    '{"findings": [{"file": "...", "line": 0, "quote": "...", "type": "...", '
    '"title": "...", "confidence": 0.0}]}',
).replace(
    "- `type` must be one of the kinds listed above.",
    """\
- `quote` is the code on that line, copied exactly as printed -- without the \
line number or the `+`/`-` mark, and without the surrounding lines. Copy it; do \
not retype it from memory. If the defect is a line the change *deleted*, quote \
that deleted line.
- `type` must be one of the kinds listed above.""",
)

# v5 covers the code that is not the library. Measured against v4 it is the best
# version on every metric, but not for the reason it was written: it gained
# exactly one finding and lost none, and that finding is in an example file --
# the clause narrowed below, not the section added above. Four of the five
# defects in test files are still met with silence. The section stays because it
# costs nothing and the hypothesis is unfalsified rather than refuted at this
# corpus size, but the measured gain belongs to the two narrowed clauses, and
# the pattern holds across the whole session: every real improvement came from
# removing something the prompt said that was wrong, never from adding
# something new.
#
# The original evidence: of the
# twenty-five findings, the model stayed silent on six of the seven whose defect
# lives in a test, a fixture, an example or a packaging script -- 86% -- against
# six of seventeen in library code. Two clauses explain it. "A defect is a place
# where the new code will behave wrong" does not describe a test that cannot run
# on a version the project supports, and the suppression list named "a
# placeholder in an example file", which reads as *the file* rather than *the
# placeholder*, and "a broad except", which is the whole of one F.4 label.
#
# So this is the same repair v3 made to the taxonomy, applied to where the
# detector looks rather than to what it calls things.

SUPPORT_CODE = """\
Tests, fixtures, examples and packaging are part of the change and defects in \
them count. A test is not exempt for being a test:

- it cannot run where the project still runs -- it needs a package this branch \
does not depend on, or a library feature older versions lack;
- it never really runs -- a skip condition that is always true, a fixture that \
returns before it asserts, a parametrisation with no values;
- it passes on the wrong thing -- it asserts against a value it also computed, \
or checks something other than what it names;
- a fixture or `conftest` guard swallows failures the tests were meant to show.

The same holds for an example script and for `setup.py`: an example that \
misuses the API teaches the misuse, and a packaging change that breaks the \
documented way of running the tests breaks it for everyone.
"""

INSTRUCTIONS_V5 = INSTRUCTIONS_V4.replace(
    "## When not to report",
    SUPPORT_CODE + "\n## When not to report",
).replace(
    """\
- Code that only looks dangerous. String building with values from a closed set, \
a placeholder in an example file, a broad `except` that re-raises.""",
    """\
- Code that only looks dangerous. String building with values from a closed set, \
a placeholder value standing in for real input, an `except` that re-raises. This \
is about the expression, not the file it is in -- and an `except` that swallows \
rather than re-raises is a defect, not a false alarm.""",
)

# v6 widens what counts as a defect. Measured on halka_bench: the model reported
# *nothing at all* on all seven pull requests it missed -- not a wrong line, no
# line -- and five of the seven are kinds its own definition excludes. The
# instructions say a defect is "a place where the new code will behave wrong",
# and a misleading name, a duplicated config value, a duplicated test block, a
# query in a loop and the same computation done twice all behave correctly. The
# suppression list then names "refactoring with no behaviour change" and
# "anything you would raise as a preference", which reads as the rest of them.
# The remaining two live in a requirements file and a settings tuple, which do
# not read as code at all.
#
# halka_bench's taxonomy spans correctness, maintainability, performance, naming
# and configuration. The definition was written for one of those five.

DEFECT_SCOPE = """\
A defect is a place this change makes worse and a reviewer would ask to fix. \
Most are about behaviour: the new code will do the wrong thing when it runs. \
Not all of them are, and these count too:

- it will behave wrong *later*, when something else changes -- a value copied \
away from the single source that documents it, a block duplicated so that one \
contract now has to be remembered in two places;
- it does the right thing wastefully -- the same expensive call twice in one \
flow, or one query per row where a batch call already exists;
- it is named against the repository's own convention, so callers read it \
wrongly even though it runs correctly;
- it is in configuration, packaging or a dependency list rather than in code -- \
a package moved out of the runtime requirements, a host dropped from an \
allow-list the code still reaches for.

Formatting and taste are still not defects, and neither is code that is merely \
unusual or unfinished.
"""

INSTRUCTIONS_V6 = INSTRUCTIONS_V5.replace(
    """\
Read the change against what the title says it does. A defect is a place where \
the new code will behave wrong -- not code that is merely unusual, unfinished or \
differently styled.
""",
    "Read the change against what the title says it does.\n\n" + DEFECT_SCOPE,
).replace(
    "- Refactoring with no behaviour change, and removal of code nothing calls. But",
    "- Refactoring that leaves one copy of the logic, and removal of code nothing "
    "calls. Duplicating a block is not refactoring. But",
).replace(
    "- Anything you would raise as a preference rather than a defect. A guard is not",
    "- Anything you would raise as a preference rather than a defect -- but a rule "
    "this repository already follows in several places is not a preference, it is "
    "a contract. A guard is not",
)

# v7 changes only the order the answer is produced in. Measured in the challenge
# stage: constrained decoding emits schema properties in order, `think: false`
# leaves no other room to work, and asking for the verdict before the reason had
# the model vote and then justify -- eleven points of F1. The detector had the
# same fault and worse: `quote` was emitted *last*, after the file, the line, the
# type, the title and the confidence, so the line it claims to accuse was chosen
# after the accusation. v7 reverses it -- copy a line, say what breaks, name the
# kind, then locate it.

INSTRUCTIONS_V7 = INSTRUCTIONS_V6.replace(
    '{"findings": [{"file": "...", "line": 0, "quote": "...", "type": "...", '
    '"title": "...", "confidence": 0.0}]}',
    '{"findings": [{"quote": "...", "title": "...", "type": "...", '
    '"file": "...", "line": 0, "confidence": 0.0}]}',
).replace(
    """\
- `file` is the path exactly as printed in a FILE header.
- `line` is a line number printed in the code. Point at the line where the \
defect is; anywhere inside the affected function counts, so prefer the most \
specific line you can name.
- `quote` is the code on that line, copied exactly as printed -- without the \
line number or the `+`/`-` mark, and without the surrounding lines. Copy it; do \
not retype it from memory. If the defect is a line the change *deleted*, quote \
that deleted line.
- `type` must be one of the kinds listed above.
- `title` is one short clause naming the problem -- under twelve words, no \
explanation, no suggested fix.""",
    """\
Answer the fields in the order they are written above; each one is meant to be \
settled before the next.

- `quote` comes first: the code you are accusing, copied exactly as printed -- \
without the line number or the `+`/`-` mark, and without the surrounding lines. \
Copy it from the excerpt; do not retype it from memory. If the defect is a line \
the change *deleted*, quote that deleted line. Choose it by reading, before you \
have decided what is wrong.
- `title` is one short clause naming what that line gets wrong -- under twelve \
words, no explanation, no suggested fix.
- `type` must be one of the kinds listed above, and must fit the title you just \
wrote.
- `file` is the path exactly as printed in a FILE header.
- `line` is the number printed against the line you quoted.""",
)

# v8 keeps v7's idea and fixes what it got wrong. v7 led with the quote and lost
# six findings: a quote is a copy, not room to reason, so asking for it first
# only makes the model commit to a line before it knows what is wrong, and it
# answers only when already certain. The challenge stage won by putting the
# *reason* before the vote; the detector's reason is the title.

INSTRUCTIONS_V8 = INSTRUCTIONS_V7.replace(
    '{"findings": [{"quote": "...", "title": "...", "type": "...", '
    '"file": "...", "line": 0, "confidence": 0.0}]}',
    '{"findings": [{"title": "...", "quote": "...", "type": "...", '
    '"file": "...", "line": 0, "confidence": 0.0}]}',
).replace(
    """\
- `quote` comes first: the code you are accusing, copied exactly as printed -- \
without the line number or the `+`/`-` mark, and without the surrounding lines. \
Copy it from the excerpt; do not retype it from memory. If the defect is a line \
the change *deleted*, quote that deleted line. Choose it by reading, before you \
have decided what is wrong.
- `title` is one short clause naming what that line gets wrong -- under twelve \
words, no explanation, no suggested fix.""",
    """\
- `title` comes first and is where you work: one short clause naming what goes \
wrong -- under twelve words, no explanation, no suggested fix.
- `quote` is the code that title is about, copied exactly as printed -- without \
the line number or the `+`/`-` mark, and without the surrounding lines. Copy it \
from the excerpt; do not retype it from memory. If the defect is a line the \
change *deleted*, quote that deleted line. If you cannot find a line that shows \
what your title says, you do not have a finding.""",
)

# --- [4a] Taksonomisiz dedektör ---------------------------------------------
#
# Every prompt version so far hands the model the list of kinds and asks it to
# find and name in one answer. This one drops the list: report what is wrong in
# your own words, and let a second call pick the name. It exists because the
# per-repository cost of a catalogue is the objection nothing else has answered
# -- a shared one costs five to ten points of F1, and writing a grounded one
# means knowing the repository before the tool is useful.

INSTRUCTIONS_OPEN = INSTRUCTIONS_V6.replace(
    "- `type` must be one of the kinds listed above, and must fit the title you "
    "just wrote.\n", ""
).replace(
    '{"findings": [{"file": "...", "line": 0, "quote": "...", "type": "...", '
    '"title": "...", "confidence": 0.0}]}',
    '{"findings": [{"file": "...", "line": 0, "quote": "...", "title": "...", '
    '"confidence": 0.0}]}'
).replace(
    "- `type` must be one of the kinds listed above.\n", ""
).replace(
    """\
- `title` is one short clause naming the problem -- under twelve words, no \
explanation, no suggested fix.""",
    """\
- `title` is one short clause naming the problem in your own words -- under \
twelve words, no explanation, no suggested fix. There is no list of kinds to \
choose from; say what goes wrong and let the words be yours.""",
)

# --- Melez: kapsam prompt'ta, isimler aramada -------------------------------
#
# Taking the catalogue out of the detector cost eight of forty-three findings.
# The list was doing two jobs and only one of them was naming: it also said what
# counts as a defect at all, which is why widening that definition in v6 helped
# and why removing it hurt. A rule list ("this repository does X") sends the
# model hunting and cost 0.067 of F1; a list of defect *classes* is a scope
# statement and earns its place.
#
# So the scope stays in front of the code, as a dozen families that name no
# repository and no product's rule ids, and the fifty-four precise names move to
# the search where breadth is free. The detector still answers in its own words:
# a family is not a label, it is the boundary of what to report.

FAMILIES = """\
Report a defect that falls in one of these families. They are broad on purpose \
-- you are not choosing a label here, only judging whether what you found is the \
kind of thing worth a comment.

- **Access.** An operation is reachable by someone it should not be, or a query \
returns data belonging to someone else.
- **Untrusted input.** A value from outside reaches something that acts on it: \
query text, a command, a response body, a decoder, an address to fetch.
- **Secrets.** A credential is written where it can be read back, or an internal \
detail reaches the client.
- **Wrong logic.** The code runs and does the wrong thing for input it will see: \
a wrong argument, unit, state, source or bound; a value clobbered instead of \
merged.
- **Hidden failure.** A failure is swallowed, reported as success, or caught so \
broadly that unrelated ones disappear with it.
- **Concurrency and repetition.** Interleaved or retried work leaves inconsistent \
state: a missing lock, an ordering nobody establishes, a duplicate guard that \
never matches.
- **Resources.** Something opened is not reliably released.
- **Wasted work.** The same expensive thing is done twice, or once per row where \
one call would do.
- **Maintenance.** A block, a value or a test is duplicated; code is unreachable \
or unreferenced; a name says something the body does not do.
- **Configuration and packaging.** A dependency, host or setting the code still \
uses is gone, or is defined twice so the two can drift.
- **Tests.** A test asserts nothing, always skips, or checks something other than \
what it names.
- **Version assumptions.** The code assumes a language or library version the \
project still has to run under.
"""

INSTRUCTIONS_HYBRID = INSTRUCTIONS_OPEN.replace(
    "## How to look", FAMILIES + "\n## How to look", 1
)

# review/v9-pr: the evidence boundary is the pull request.
#
# The broad catalogue stays -- it names defects a repository may not contain,
# on purpose -- but every definition is rewritten so that it can be decided
# from the code the model is shown. The v6-broad definitions compare against
# what the pack does not carry: "weaker than its siblings", "used elsewhere",
# "the convention its neighbours follow". Measured on the best chain, nine of
# its eleven remaining false alarms were claims whose truth hangs on such an
# unseen comparison, and two model families flagged the same eight halka
# spots. A real repository has no clean twin to calibrate against and no
# promise that the sibling is in the diff. So where a comparison is part of a
# kind, the other side of it has to be in this change too.
#
# MEASURED: halka lost, zincir gained, so it does not pass. Same chain, same
# build, same names as v6-broad. halka 44/9/4 became 40/14/8, F1 0.871 to
# 0.784. zincir 7/6/9 became 7/5/9, F1 0.483 to 0.500 -- still under v6-shared's
# 8/3/8. On halka the five false alarms whose truth hung on an unseen rule did
# go (conc-01, conc-02, conc-03, corr-01, err-01), and three of those clean
# twins were flagged again at the same place under another name: conc-01-temiz
# line 77 as wrong_state_check, conc-02-temiz as missing_authz_check, err-01-temiz
# as unvalidated_passthrough. The definitions decide what the model calls the
# spot; they do not decide whether it speaks there. Four labels were lost, and
# not only ones resting on a repository rule: grounding c, a, b and c.
PR_BOUNDARY = """\
Judge only from what you are shown: the title and the code this pull request changed. A defect has to be established by those lines. If deciding it would need something you are not shown -- how other modules do it, what the rest of the repository treats as a rule, what a caller outside the change passes -- you cannot establish it, so do not report it.
"""


def _pr_instructions() -> str:
    text = INSTRUCTIONS_V6
    edits = (
        ("Read the change against what the title says it does.\n\n",
         "Read the change against what the title says it does.\n\n" + PR_BOUNDARY + "\n"),
        ("a value copied away from the single source that documents it, a block duplicated",
         "a value defined a second time, a block duplicated"),
        ("or one query per row where a batch call already exists;",
         "or one query per row inside a loop;"),
        ("it is named against the repository's own convention, so callers read it wrongly",
         "its name says something its body does not do, so callers read it wrongly"),
        ("- Anything you would raise as a preference rather than a defect -- but a rule this "
         "repository already follows in several places is not a preference, it is a contract. "
         "A guard is not a preference: a check the change applies on one path and not on its "
         "sibling,",
         "- Anything you would raise as a preference rather than a defect. A guard is not a "
         "preference: a check the change applies on one path and leaves off another path in "
         "the same change,"),
    )
    for old, new in edits:
        if text.count(old) != 1:
            raise AssertionError(f"v9-pr edit does not apply exactly once: {old[:60]!r}")
        text = text.replace(old, new, 1)
    return text


INSTRUCTIONS_PR = _pr_instructions()

PR_TYPES_HALKA = {
    "missing_authz_check": "An operation that changes or reveals data is reachable without any permission check, or the change checks permission on one path it adds and not on another path it adds for the same operation.",
    "crossfile_ownership": "A query in the change reads or writes rows without restricting them to the tenant or owner the request belongs to, although that tenant or owner is in scope where the query is built.",
    "crossfile_data_exposure": "A response built in the change includes a field that is sensitive by nature -- a password hash, a token, a secret, internal notes -- or serializes a whole stored object instead of choosing its fields.",
    "crossfile_error_propagation": "A failure raised or returned by a call in the change is caught or ignored, and the caller goes on to report success.",
    "crossfile_idempotency": "The key a duplicate guard uses includes something that changes on every run -- an attempt count, a timestamp, a random value -- so the same event never matches itself.",
    "crossfile_ordering": "Code in the change consumes items as if they arrive in a particular order, and nothing in the code shown establishes that order.",
    "crossfile_unit_mismatch": "A value is passed in one unit where the receiving code shown uses it in another -- seconds for milliseconds, minor units for major, bytes for kilobytes.",
    "sql_injection": "A runtime value is placed in the query text rather than in the parameters the driver binds.",
    "command_injection": "A runtime value is placed in the command text rather than in an element of an argument list.",
    "xss": "A value from a request or a user reaches an HTML response without escaping.",
    "error_detail_disclosure": "An exception message or a stack trace is placed into a response sent to the client.",
    "hardcoded_credential": "A password, key, token or other secret is written as a literal in the source.",
    "weak_crypto_primitive": "A broken or unsuitable primitive is used for a security purpose: a broken digest for passwords or signatures, ordinary randomness for a secret, or a comparison of secrets that returns early.",
    "unvalidated_passthrough": "Request data is handed to a call that stores, executes or forwards it, with no validation or restriction applied to it in the code shown.",
    "wrong_argument": "Arguments are passed in an order or a role that does not match the parameters of the callee shown.",
    "wrong_data_source": "A value that has to be current is read from a copy that can lag -- a cache, a snapshot, a denormalized field -- while the authoritative value is available in the same code.",
    "wrong_state_check": "A state is compared against a value the code shown never assigns to it, so the branch is never taken or always taken.",
    "silent_overwrite": "A write replaces a whole stored structure where the code meant to change part of it, discarding the fields it did not set.",
    "unreachable_code": "The control flow above these statements always leaves before them.",
    "unused_symbol": "The change defines a local variable, a module-private name or an import that nothing uses.",
    "misleading_name": "The name says something the body does not do -- it promises one item and returns a list, says it checks and it modifies, says it is safe and it is not.",
    "duplicated_block": "The change adds a block whose logic repeats another block in the code shown, instead of calling it.",
    "duplicated_config": "The change defines the same configuration value in two places in the code shown, so the two can drift apart.",
    "duplicated_test_block": "The change adds test assertions that repeat another test's assertions in the code shown, instead of sharing them.",
    "broad_except": "The clause catches a type wider than anything the body it guards can raise.",
    "swallowed_exception": "An exception is caught and then neither logged, re-raised nor reported to the caller.",
    "unclosed_resource": "A file, connection or similar resource is opened without a `with` block or a `close` on every path.",
    "redundant_work": "The same value is computed twice in one flow where the first result is still in scope.",
    "work_in_loop": "A query or remote call is made once per item inside a loop over a collection that could be fetched in one call.",
    "removed_dependency": "A package is removed from the runtime requirements while code in the change still imports it.",
    "removed_network_config": "A host, URL or endpoint is removed from configuration while code in the change still uses it.",
    "missing_assertion": "A test runs the code under test and asserts nothing about its result, or asserts only something that holds whatever the code does.",
    "hardcoded_endpoint": "A test writes a request path as a literal string instead of building it with the framework's URL resolution.",
    "null_deref": "The result of a call that can return None is used without a check.",
    "off_by_one": "An index, bound or page offset is one away from what it should be -- a range that skips the first or last item, a limit that allows one too many.",
    "unguarded_dict_access": "A key that comes from outside -- a request, a payload, a file -- indexes a mapping directly, so a missing key raises.",
    "missing_lock": "Shared state is read, changed and written back without a lock or a transaction, so two concurrent runs can lose an update.",
    "secret_in_log": "A value obtained as a secret is written to a log in clear.",
    "ssrf_unvalidated_fetch": "A URL or host taken from the request is fetched by the server without restricting where it may point.",
    "unsafe_deserialization": "Data from outside is decoded with a loader that can construct arbitrary objects -- pickle, unsafe YAML, eval.",
    "divergent_change": "One module is changed for reasons that have nothing to do with each other.",
    "path_traversal": "A file path is built from a value the caller controls and opened without checking that it stays inside the intended directory.",
    "open_redirect": "A redirect is sent to a location taken from the request without restricting it to trusted targets.",
    "mass_assignment": "A request body is bound wholesale onto a stored object, so fields that should not be writable from outside can be set.",
    "insecure_default": "A default turns a protection off -- verification skipped, debug left on, a permissive origin.",
    "unbounded_resource": "A response, file or query result is read whole into memory with no size limit.",
    "float_money": "A monetary amount is held or computed in a floating-point type.",
    "naive_datetime": "A timestamp is created or compared without a timezone.",
    "blocking_call_in_async": "A call that blocks -- a socket, a file, a sleep -- runs inside a coroutine.",
    "mutable_default_argument": "A parameter defaults to a mutable object, so the value outlives the call and is shared by every later one.",
    "regex_denial_of_service": "A pattern with nested or overlapping repetition is matched against input from outside, where the time it takes grows faster than the input does.",
    "breaking_public_api": "A name callers outside this change may depend on -- an exported symbol, a flag, an attribute a subclass overrides -- is removed, renamed or narrowed.",
    "missing_migration": "A stored model's fields are changed and the change contains no migration for it.",
}
PR_ADDITIONS = {
    "wrong_assertion_target": "A test asserts on something other than the effect it names -- the input it just built, a mock's own return value, or a field the code under test never writes -- so it passes whatever that code does.",
    "leaky_test_state": "A test leaves process state behind, or depends on state another test left, so its result depends on which tests ran before it.",
    "unsafe_default": "A default value turns a protection off or removes a bound -- verification skipped, a timeout unlimited, a debug path left on.",
    "contradictory_setting": "Two settings that are read together state incompatible things, so one of them cannot take effect.",
    "removed_config_key": "A configuration key is deleted while code in the change still reads it, so the reader falls back to a value nobody chose.",
}
# Same names, same order as v6-broad on each corpus: only the definitions move.
PR_TYPES_HALKA = {name: PR_TYPES_HALKA[name] for name in HALKA_TYPES_BROAD}
_PR_ZINCIR = {**PR_TYPES_HALKA, **PR_ADDITIONS}
TAXONOMIES_PR = {**TAXONOMIES_V3, "halka": PR_TYPES_HALKA,
                 "zincir": {name: _PR_ZINCIR[name] for name in TAXONOMIES_BROAD["zincir"]}}


# The typed versions: each file is read by its role. Section D14 found the
# pack "shows every file the same way and asks all of them one question" while
# the model's power to tell a defective file from its clean twin differs sharply
# by role; stage [4c] tried the obvious repair -- a second call per file, with a
# role question -- and it lowered the bar on both twins alike. These keep the
# single call and the whole pull request in view, and let the pack say what
# each file is. Application code is told to read as before, so halka, which is
# almost all application code, is moved as little as the idea allows: whatever
# changes should change in test files and modules of constants.
#
# Each is its control plus one section and nothing else, so a difference in the
# measurement is a difference this section made.
#
# MEASURED, AND BOTH LOST. Same chain as the controls (deletions, [4b], the
# quote gate's output fed to challenge and regate). zincir_dev 8/3/8 became
# 8/5/8, F1 0.593 to 0.552, balanced accuracy 91.0% to 85.4%: raw reports on
# clean test files went from none to two, and they are the two [4c] produced
# -- wrong_assertion_target on bus-01-temiz's test file, leaky_test_state on
# clean-counters-many. halka 44/9/4 became 42/9/6, F1 0.871 to 0.849, losing
# conf-03 and perf-02, both in application code, which the section tells the
# model to read as before. Saying so did not hold it still: three hundred
# tokens of role guidance moved the whole review, not the files it named.
# Telling the model what a file is, in the call or after it, lowers its bar
# for that kind of file on both twins; it does not teach it the defect.
ROLE_SECTION = """\
## Read each file by its role

Every `# FILE` header is followed by a line naming the file's role. The rules in \
this prompt apply to every file; the role says what a defect looks like in that \
kind of file, and what does not count as one.

- **Application code.** Read it exactly as the rest of this prompt describes.
- **Test file.** A test is judged by what it proves about the code it exercises. \
It is defective when it would still pass with that behaviour broken, when it \
checks something other than what its name promises, or when it leaves state \
behind that another test reads -- and a deleted assertion or reset counts. A \
test that checks less than it could is not defective for that; the question is \
whether what it does claim can fail.
- **Module of constants.** A constant is judged by the code that reads it. It is \
defective when its value is unsafe for that code, when it contradicts another \
value it has to agree with, or when the change removed or renamed a name that \
code still reads. A value you would merely have chosen differently is not a \
defect: if you cannot name the code that reads it and what goes wrong there, \
do not report it.

"""

INSTRUCTIONS_V6_TYPED = INSTRUCTIONS_V6.replace(
    "## When not to report", ROLE_SECTION + "## When not to report", 1)

VERSIONS = {
    "review/v1": (INSTRUCTIONS_V1, TAXONOMIES),
    "review/v2": (INSTRUCTIONS_V2, TAXONOMIES),
    "review/v3": (INSTRUCTIONS_V3, TAXONOMIES_V3),
    "review/v4": (INSTRUCTIONS_V4, TAXONOMIES_V3),
    "review/v5": (INSTRUCTIONS_V5, TAXONOMIES_V3),
    "review/v6": (INSTRUCTIONS_V6, TAXONOMIES_V3),
    "review/v7": (INSTRUCTIONS_V7, TAXONOMIES_V3),
    "review/v8": (INSTRUCTIONS_V8, TAXONOMIES_V3),
    "review/v6-generic": (INSTRUCTIONS_V6, TAXONOMIES_GENERIC),
    "review/v6-wide": (INSTRUCTIONS_V6, TAXONOMIES_WIDE),
    "review/v6-evidence": (INSTRUCTIONS_V6, TAXONOMIES_EVIDENCE),
    "review/v6-located": (INSTRUCTIONS_V6, TAXONOMIES_LOCATED),
    "review/v6-reachable": (INSTRUCTIONS_V6, TAXONOMIES_REACHABLE),
    "review/v6-demo-evidence": (INSTRUCTIONS_V6, TAXONOMIES_DEMO_EVIDENCE),
    "review/open": (INSTRUCTIONS_OPEN, TAXONOMIES_V3),
    "review/hybrid": (INSTRUCTIONS_HYBRID, TAXONOMIES_V3),
    "review/v6-wide-evidence": (INSTRUCTIONS_V6, TAXONOMIES_WIDE_EVIDENCE),
    "review/v6-broad": (INSTRUCTIONS_V6, TAXONOMIES_BROAD),
    "review/v6-shared": (INSTRUCTIONS_V6, TAXONOMIES_SHARED),
    "review/v6-broad-typed": (INSTRUCTIONS_V6_TYPED, TAXONOMIES_BROAD),
    "review/v6-shared-typed": (INSTRUCTIONS_V6_TYPED, TAXONOMIES_SHARED),
    "review/v9-pr": (INSTRUCTIONS_PR, TAXONOMIES_PR),
}

# Which versions print each file's role under its header; the pack reads this.
TYPED = frozenset({"review/v6-broad-typed", "review/v6-shared-typed"})

# Versions that carry no catalogue: the detector says what is wrong in its own
# words and stage [4b] maps that onto a name. Every measurement behind the
# taxonomy work says this is worth trying -- breadth in the prompt manufactured
# false alarms, a rule list sent the model hunting, and naming is the closed
# question the challenge stage does well -- and none of it says the detector can
# work without a catalogue at all. That is what the run answers.
OPEN = frozenset({"review/open", "review/hybrid"})

# Which versions ask for the quote, so the schema and the filter agree without
# either of them guessing from the version string.
QUOTED = frozenset({"review/v4", "review/v5", "review/v6", "review/v7", "review/v8",
                    "review/v6-generic", "review/v6-wide", "review/v6-evidence",
                    "review/v6-located", "review/v6-reachable",
                    "review/v6-demo-evidence", "review/v6-wide-evidence",
                    "review/v6-broad", "review/v6-shared",
                    "review/v6-broad-typed", "review/v6-shared-typed", "review/v9-pr"})

# Which versions want the answer produced evidence-first. Everything measured
# before v7 was measured under the legacy order and has to stay on it.
EVIDENCE_FIRST = frozenset({"review/v7"})
CLAIM_FIRST = frozenset({"review/v8"})


def types(dataset: str, version: str | None = None) -> list[str]:
    """The names the schema constrains the answer to.

    Version-aware: two experiments change the *names*, not only their wording,
    and a schema built from the default map would let the model answer with a
    type the prompt never listed.
    """
    if dataset not in TAXONOMIES:
        raise KeyError(f"unknown dataset {dataset!r}; have {', '.join(TAXONOMIES)}")
    taxonomies = VERSIONS[version][1] if version in VERSIONS else TAXONOMIES
    if dataset not in taxonomies:
        # Section 0.5. A variant that covers one corpus and silently hands every
        # other the default list is the failure this check exists to stop: the
        # run is labelled with the variant's name and is the default underneath.
        raise KeyError(f"prompt version {version!r} defines no catalogue for {dataset!r}; "
                       f"it covers {', '.join(sorted(taxonomies))}")
    return list(taxonomies[dataset])


def system(dataset: str, version: str = PROMPT_VERSION) -> str:
    """The full system prompt: the shared instructions plus this dataset's kinds."""
    if version not in VERSIONS:
        raise KeyError(f"unknown prompt version {version!r}; have {', '.join(VERSIONS)}")
    instructions, taxonomies = VERSIONS[version]
    head, tail = instructions.split("## How to look")
    if version in OPEN:
        # No catalogue at all. The whole point of an open version is that naming
        # is somebody else's call, so listing kinds here would put the breadth
        # back in the one prompt it was taken out of.
        return f"{head.format(format=FORMATS[dataset])}## How to look{tail}"
    if dataset not in taxonomies:
        raise KeyError(f"prompt version {version!r} defines no catalogue for {dataset!r}; "
                       f"it covers {', '.join(sorted(taxonomies))}")
    catalogue = "\n".join(f"- `{name}` -- {text}" for name, text in taxonomies[dataset].items())
    return (
        f"{head.format(format=FORMATS[dataset])}"
        f"## The kinds of defect you report\n\nReport only these, and nothing else:\n\n"
        f"{catalogue}\n\n"
        f"## How to look{tail}"
    )
