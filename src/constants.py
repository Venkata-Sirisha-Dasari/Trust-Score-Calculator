"""Constants and threshold configuration for the trust scoring engine.

Keeping these values in one module means a threshold change (e.g. tightening
the SAFETY_RISK bar after a policy review) happens in exactly one place.
"""

import json
from pathlib import Path

EVALUATOR_VERSION = "1.0.0"

# The JSON Schema file is the single source of truth for which dimensions
# are valid, and for score/weight bounds. VALID_DIMENSIONS is *derived* from
# it below rather than hand-maintained separately, so the schema and the
# Python code can never drift out of sync with each other.
_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "trust_schema.json"
with open(_SCHEMA_PATH, encoding="utf-8") as _f:
    INPUT_SCHEMA: dict = json.load(_f)

VALID_DIMENSIONS: set[str] = set(INPUT_SCHEMA["properties"]["scores"]["propertyNames"]["enum"])

# Score bounds (inclusive). 0.0 = worst, 1.0 = best.
SCORE_MIN: float = 0.0
SCORE_MAX: float = 1.0

# Note: weights must be non-negative (a negative weight would invert a
# dimension's contribution, which has no valid governance meaning), but
# that rule now lives only in schema/trust_schema.json ("minimum": 0.0)
# rather than as a separate Python constant, to avoid the two drifting.

# A weight above this is very likely a data-entry mistake (e.g. entering
# "5" meaning "50%" instead of 0.5). We still accept it, but warn.
WEIGHT_LARGE_WARN: float = 10.0

# A score above this is almost certainly a data pipeline error (e.g. a raw
# percentage like 85 was passed instead of 0.85). Checked before the normal
# range check purely so the error message can be more specific about *why*
# the value is likely wrong.
SCORE_MAX_ALLOWED: float = 1e6

# Tolerance for floating-point "sums to 1.0 / 0.0" comparisons.
FLOAT_TOLERANCE: float = 1e-9

# Trust-score risk tiers. Boundaries are inclusive on the lower risk side
# (see TrustScoreCalculator.classify_risk for the exact comparison logic
# and rationale for treating boundary values conservatively).
RISK_THRESHOLDS = {
    "CRITICAL": 0.40,
    "HIGH": 0.60,
    "MEDIUM": 0.75,
}

# Per-dimension flag thresholds. Deliberately stricter than the aggregate
# score thresholds above: a single failed dimension (e.g. SAFETY at 0.30)
# is a categorical governance risk even if other dimensions pull the
# aggregate score up to MEDIUM.
DIMENSION_FLAG_THRESHOLDS = {
    "accuracy": 0.60,
    "robustness": 0.60,
    "fairness": 0.55,
    "safety": 0.50,
    "privacy": 0.55,
    "transparency": 0.55,
}
