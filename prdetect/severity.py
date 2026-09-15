"""How serious each kind of defect is: HIGH, MEDIUM or LOW.

Severity belongs to the kind, not to the claim, so it is read from this table
after a finding is published and never changes what the detector is asked or
what it finds.

HIGH    a hole someone can use, or a loss of data, money or integrity;
MEDIUM  code that does the wrong thing when it runs;
LOW     what a later reader or the test suite pays for.
"""
from __future__ import annotations

HIGH, MEDIUM, LOW = "HIGH", "MEDIUM", "LOW"
ORDER = (HIGH, MEDIUM, LOW)

SEVERITY = {
    # Security, credentials, and integrity of stored data or money.
    "missing_authz_check": HIGH,
    "crossfile_ownership": HIGH,
    "crossfile_data_exposure": HIGH,
    "crossfile_idempotency": HIGH,
    "sql_injection": HIGH,
    "command_injection": HIGH,
    "xss": HIGH,
    "hardcoded_credential": HIGH,
    "weak_crypto_primitive": HIGH,
    "unvalidated_passthrough": HIGH,
    "secret_in_log": HIGH,
    "ssrf_unvalidated_fetch": HIGH,
    "unsafe_deserialization": HIGH,
    "path_traversal": HIGH,
    "mass_assignment": HIGH,
    "dynamic_code_execution": HIGH,
    "insecure_transport": HIGH,
    "insecure_randomness": HIGH,
    "overly_permissive_permission": HIGH,
    "missing_transaction": HIGH,
    "check_then_act_race": HIGH,
    "missing_duplicate_guard": HIGH,
    # Wrong behaviour at run time.
    "crossfile_error_propagation": MEDIUM,
    "crossfile_ordering": MEDIUM,
    "crossfile_unit_mismatch": MEDIUM,
    "error_detail_disclosure": MEDIUM,
    "open_redirect": MEDIUM,
    "wrong_argument": MEDIUM,
    "wrong_data_source": MEDIUM,
    "wrong_state_check": MEDIUM,
    "silent_overwrite": MEDIUM,
    "swallowed_exception": MEDIUM,
    "unclosed_resource": MEDIUM,
    "work_in_loop": MEDIUM,
    "removed_dependency": MEDIUM,
    "null_deref": MEDIUM,
    "off_by_one": MEDIUM,
    "unguarded_dict_access": MEDIUM,
    "missing_lock": MEDIUM,
    "unbounded_resource": MEDIUM,
    "float_money": MEDIUM,
    "naive_datetime": MEDIUM,
    "blocking_call_in_async": MEDIUM,
    "mutable_default_argument": MEDIUM,
    "regex_denial_of_service": MEDIUM,
    "breaking_public_api": MEDIUM,
    "missing_migration": MEDIUM,
    "unsafe_default": MEDIUM,
    "contradictory_setting": MEDIUM,
    "removed_config_key": MEDIUM,
    "removed_network_config": MEDIUM,
    "missing_timeout": MEDIUM,
    "retry_without_bound": MEDIUM,
    "unawaited_coroutine": MEDIUM,
    "global_state_mutation": MEDIUM,
    "lossy_conversion": MEDIUM,
    "encoding_assumption": MEDIUM,
    "unsafe_temp_file": MEDIUM,
    "ignored_return_value": MEDIUM,
    "stale_cache_write": MEDIUM,
    "loop_without_progress": MEDIUM,
    "business_logic": MEDIUM,
    # Maintainability and tests.
    "unreachable_code": LOW,
    "unused_symbol": LOW,
    "misleading_name": LOW,
    "duplicated_block": LOW,
    "duplicated_config": LOW,
    "duplicated_test_block": LOW,
    "broad_except": LOW,
    "redundant_work": LOW,
    "divergent_change": LOW,
    "missing_assertion": LOW,
    "hardcoded_endpoint": LOW,
    "wrong_assertion_target": LOW,
    "leaky_test_state": LOW,
    "disabled_test": LOW,
}


def severity(kind: str) -> str:
    """The severity of a kind; MEDIUM for a kind the table does not know."""
    return SEVERITY.get(kind, MEDIUM)
