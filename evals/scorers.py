"""
Scorers for the eval harness. Each takes the model's output plus the case's
`expected` value and returns (passed: bool, detail: str).

Kept deliberately simple and dependency-free — the point of v1 is to prove the
run -> score -> record loop. Richer scorers (LLM-as-judge, regex, numeric
tolerance) slot in here later without touching the runner.
"""

import json


def contains(output: str, expected: str) -> tuple[bool, str]:
    ok = expected in output
    return ok, f"expected substring {expected!r}"


def contains_ci(output: str, expected: str) -> tuple[bool, str]:
    ok = expected.lower() in output.lower()
    return ok, f"expected substring (ci) {expected!r}"


def equals_ci(output: str, expected: str) -> tuple[bool, str]:
    ok = output.strip().lower() == expected.strip().lower()
    return ok, f"expected exact (ci) {expected!r}"


def _strip_code_fence(text: str) -> str:
    """Drop a leading ```json / ``` fence and trailing ``` that models often add."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def json_valid(output: str, expected) -> tuple[bool, str]:
    # Tolerate markdown code fences — models wrap JSON in ```json ... ``` even when
    # told not to. We care that the payload is valid JSON, not about the fence.
    try:
        json.loads(_strip_code_fence(output))
        return True, "valid JSON"
    except Exception as e:
        return False, f"invalid JSON: {e}"


SCORERS = {
    "contains": contains,
    "contains_ci": contains_ci,
    "equals_ci": equals_ci,
    "json_valid": json_valid,
}


def score(scorer_name: str, output: str, expected) -> tuple[bool, str]:
    fn = SCORERS.get(scorer_name)
    if fn is None:
        raise ValueError(f"Unknown scorer: '{scorer_name}'. Available: {list(SCORERS)}")
    return fn(output, expected)
