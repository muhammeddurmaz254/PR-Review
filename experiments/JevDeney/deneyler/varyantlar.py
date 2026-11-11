"""Deney-1 varyantlarının istek gövdeleri ve soru metinleri (tek yerde).

Soru metinleri 2026-11-10'da, hiçbir Jev sonucu görülmeden sabitlendi.
Benchmark tür adları ya da örnekleri kullanılmaz (bkz. TASARIM.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from istemci import SORULAR  # noqa: E402

_ODAK = ("Judge only what this change adds or changes. "
         "Assume the code before the change was correct.")


def _noul(soru: str, dogru: str, yanlis: str, yanlis_degil: str | None = None) -> dict:
    false = {"what": yanlis}
    if yanlis_degil:
        false["not_for"] = yanlis_degil
    return {"type": "noul",
            "instructions": {"question": soru, "focus": _ODAK},
            "criteria": {"true": {"what": dogru}, "false": false}}


# Deney-1B: kod incelemesinde yaygın kusur sınıfları, her biri tek ve gözlenebilir yargı.
ATOMIK_SORULAR: dict[str, dict] = {
    "wrong_logic": _noul(
        "Do the added or changed lines in `diff` use a wrong condition, value, variable "
        "or calculation?",
        "A changed line computes, compares or selects something other than what the "
        "surrounding code needs",
        "Every changed condition, value and calculation does what the surrounding code needs"),
    "mishandled_error": _noul(
        "Do the added or changed lines in `diff` mishandle an error or exception?",
        "A failure is ignored, swallowed, reported as success, or handled in a way that "
        "loses it",
        "Failures are raised, reported or handled as the surrounding code expects"),
    "missing_check": _noul(
        "Do the added or changed lines in `diff` skip a permission, validation or safety "
        "check that this code path needs?",
        "A code path can now run without a check that similar paths perform",
        "Every new code path performs the checks it needs",
        "A check that is present but written differently"),
    "unsafe_input_use": _noul(
        "Do the added or changed lines in `diff` pass untrusted input into a query, "
        "command, file path, URL or deserializer without protection?",
        "Data from a user or an outside system reaches such a sink unprotected",
        "Untrusted data is not used this way, or it is validated, escaped or parameterized"),
    "exposes_sensitive_data": _noul(
        "Do the added or changed lines in `diff` expose a secret or sensitive data?",
        "A credential, key, token or personal data is hard-coded, logged, returned or "
        "sent where it should not be",
        "No secret or sensitive data is written, logged, returned or sent"),
    "weakened_test": _noul(
        "Do the added or changed lines in `diff` leave a test unable to detect the failure "
        "it is meant to catch?",
        "A test asserts too little, the wrong thing, or nothing, or depends on state that "
        "makes it pass regardless",
        "Each changed test still fails when the behaviour it checks is broken",
        "Files that are not tests"),
    "broken_configuration": _noul(
        "Do the added or changed lines in `diff` set, remove or combine configuration "
        "values in a way that breaks or weakens the application?",
        "A setting is missing, contradicts another setting, or has an unsafe default",
        "Configuration values stay complete, consistent and safe",
        "Files that hold no configuration"),
    "race_condition": _noul(
        "Do the added or changed lines in `diff` introduce a race condition or unsafe "
        "concurrent access?",
        "Two executions can interleave so that data is lost, duplicated or corrupted",
        "Shared state is accessed safely or not shared at all"),
    "resource_problem": _noul(
        "Do the added or changed lines in `diff` leak a resource or use one without bound?",
        "A connection, file, task, memory or query count grows or stays open without limit",
        "Resources are released and bounded"),
}


def atomik_govde(dosya: str, parca: str, model: str) -> dict:
    return {"model": model, "state": {"file_path": dosya, "diff": parca},
            "questions": ATOMIK_SORULAR}


def pr_baglam_govde(dosya_parcalari: dict[str, str], model: str) -> tuple[dict, dict[str, str]]:
    """PR'daki bütün dosyalar tek state'te; her dosya için ana sorularla aynı iki Noul.

    Dönüş: (istek gövdesi, soru adı → dosya yolu)."""
    dosyalar, sorular, harita = {}, {}, {}
    for i, (dosya, parca) in enumerate(dosya_parcalari.items()):
        anahtar = f"f{i}"
        dosyalar[anahtar] = {"path": dosya, "diff": parca}
        for soru_adi, ana in SORULAR.items():
            kisa = soru_adi.removeprefix("introduces_")
            ad = f"{anahtar}_{kisa}"
            soru = (ana["instructions"]["question"]
                    .replace("`diff`", f"`changed_files.{anahtar}.diff`")
                    .replace("`file_path`", f"`changed_files.{anahtar}.path`"))
            sorular[ad] = {
                "type": "noul",
                "instructions": {
                    "question": soru,
                    "focus": _ODAK + " The other entries in `changed_files` belong to the "
                             "same pull request; use them as context.",
                },
                "criteria": ana["criteria"],
            }
            harita[ad] = dosya
    return {"model": model, "state": {"changed_files": dosyalar}, "questions": sorular}, harita


# Deney-1B v2 (2026-11-11). Geliştirme yalnızca halka + zincir_dev üzerinde yapıldı;
# zincir_holdout sonuçlarına bakılmadı. Değişiklikler ve gerekçeleri:
# - "veya" ile birleşmiş yargılar ayrıldı (TypeSafe: tek soruda tek yargı):
#   missing_check -> missing_authorization + missing_input_validation (v1'de 32 isabet / 35 yanlış alarm)
#   unsafe_input_use -> injection + outbound_request + unsafe_deserialization (10 / 10)
# - race_condition gözlenebilir duruma daraltıldı: shared_state_race (6 / 11)
# - v1'in kör kaldığı genel sınıflar eklendi: weak_crypto, boundary_case, inefficient_data_access
_KORUNAN = ("wrong_logic", "mishandled_error", "exposes_sensitive_data", "weakened_test",
            "broken_configuration", "resource_problem")

ATOMIK_V2_SORULAR: dict[str, dict] = {ad: ATOMIK_SORULAR[ad] for ad in _KORUNAN}
ATOMIK_V2_SORULAR.update({
    "missing_authorization": _noul(
        "Do the added or changed lines in `diff` let a user act on or read data without "
        "checking that they are allowed to?",
        "A new or changed code path skips the permission, role or ownership check that "
        "similar paths in the file perform, or uses a weaker check",
        "Every new or changed code path checks the caller's permission as strictly as "
        "similar paths do",
        "Input format or range checks, which are not permission checks"),
    "missing_input_validation": _noul(
        "Do the added or changed lines in `diff` use an input value without checking its "
        "format, type or range first?",
        "A value from a request, file, message or argument is used although it can be "
        "missing, malformed or out of range",
        "Input values are checked before use, or cannot be invalid at this point",
        "Permission checks, which are not input validation"),
    "injection": _noul(
        "Do the added or changed lines in `diff` build a query, shell command or template "
        "from untrusted input?",
        "Untrusted text is concatenated or formatted into a query, command or template",
        "Queries and commands use parameters or fixed text only"),
    "outbound_request": _noul(
        "Do the added or changed lines in `diff` make the server send a request to an "
        "address that untrusted input controls?",
        "A URL, host or path from a user or an outside system is fetched without an "
        "allow-list",
        "Outgoing requests go only to fixed or checked addresses"),
    "unsafe_deserialization": _noul(
        "Do the added or changed lines in `diff` deserialize untrusted data with a loader "
        "that can run code or build arbitrary objects?",
        "Untrusted bytes reach a loader such as pickle or an unsafe YAML load",
        "Untrusted data is parsed only with a safe loader such as JSON"),
    "shared_state_race": _noul(
        "Do the added or changed lines in `diff` read, change and write shared data without "
        "a lock, transaction or atomic operation?",
        "Two concurrent runs can both read the old value and one update is lost or "
        "applied twice",
        "Shared data is updated atomically, under a lock or transaction",
        "Code that only reads shared data"),
    "weak_crypto": _noul(
        "Do the added or changed lines in `diff` use cryptography in a weak or incorrect way?",
        "A weak hash or cipher, a predictable random value for a secret, or a non-constant-"
        "time comparison of secrets is used",
        "Cryptography uses strong algorithms and safe comparisons, or is not used"),
    "boundary_case": _noul(
        "Do the added or changed lines in `diff` handle an empty, missing, zero, negative "
        "or last item wrongly?",
        "An edge value makes the new code crash, skip an item or return a wrong result",
        "The new code gives correct results for edge values"),
    "inefficient_data_access": _noul(
        "Do the added or changed lines in `diff` access the database or a remote service "
        "much more often than needed?",
        "A query or remote call runs once per item in a loop, or loads far more rows than "
        "the code uses",
        "Data is fetched in a bounded number of calls"),
})


def atomik_v2_govde(dosya: str, parca: str, model: str) -> dict:
    return {"model": model, "state": {"file_path": dosya, "diff": parca},
            "questions": ATOMIK_V2_SORULAR}
