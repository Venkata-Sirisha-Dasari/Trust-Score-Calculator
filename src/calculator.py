"""Trust score evaluation engine for six AI governance dimensions.

Contains the core scoring engine: input validation, weight normalization,
weighted scoring, risk classification, and SHA-256 sealed evidence
artifact generation.
"""

import hashlib
import logging
import uuid
from datetime import datetime, timezone

import jsonschema

from .constants import (
    DIMENSION_FLAG_THRESHOLDS,
    EVALUATOR_VERSION,
    FLOAT_TOLERANCE,
    INPUT_SCHEMA,
    RISK_THRESHOLDS,
    SCORE_MAX,
    SCORE_MAX_ALLOWED,
    SCORE_MIN,
    VALID_DIMENSIONS,
    WEIGHT_LARGE_WARN,
)
from .exceptions import EvidenceError, NormalizationError, ValidationError
from .models import EvidenceArtifact, RiskFlag, RiskLevel, TrustScoreResult

logger = logging.getLogger(__name__)

# Maps dimension name -> the RiskFlag raised when that dimension underperforms.
_DIMENSION_FLAG_MAP: dict[str, RiskFlag] = {
    "accuracy": RiskFlag.ACCURACY_RISK,
    "robustness": RiskFlag.ROBUSTNESS_RISK,
    "fairness": RiskFlag.FAIRNESS_RISK,
    "safety": RiskFlag.SAFETY_RISK,
    "privacy": RiskFlag.PRIVACY_RISK,
    "transparency": RiskFlag.TRANSPARENCY_RISK,
}


class TrustScoreCalculator:
    """Evaluates an AI system across six governance dimensions.

    Produces a weighted trust score, a risk classification, explainable
    per-dimension risk flags, and a SHA-256 sealed evidence artifact
    suitable for audit or compliance review.

    Parameters
    ----------
    scores : dict[str, float]
        Dimension name -> raw score in [0.0, 1.0].
    weights : dict[str, float]
        Dimension name -> non-negative weight. Need not sum to 1.0;
        normalization is applied automatically.

    Example
    -------
    >>> calc = TrustScoreCalculator(scores, weights)
    >>> result = calc.calculate_score()
    >>> artifact = calc.generate_evidence(result)
    >>> print(artifact.to_json())
    """

    def __init__(self, scores: dict[str, float], weights: dict[str, float]) -> None:
        self._raw_weights = dict(weights)
        self.scores = dict(scores)
        self.weights = dict(weights)
        self.normalized_weights: dict[str, float] = {}
        self._warnings: list[str] = []
        logger.info("TrustScoreCalculator initialised | dimensions=%s", list(scores.keys()))

    def validate_inputs(self) -> None:
        """Validate scores and weights before any computation.

        Validation happens in two layers with distinct jobs:

        1. `INPUT_SCHEMA` (schema/trust_schema.json) is the real
           structural validator: it enforces that only the six known
           dimensions are present, that scores are numeric in [0, 1],
           and that weights are numeric and non-negative. Nothing
           below duplicates what the schema already checks.
        2. A small amount of manual logic handles what a JSON Schema
           cannot express: one specific, more actionable error message
           for a likely data-entry mistake, and warnings (as opposed
           to hard failures) for missing dimensions, missing weights,
           and suspiciously large weights.

        Raises
        ------
        ValidationError
            On any fatal problem: empty input, unknown dimension name,
            non-numeric or out-of-range score, or negative weight.
            Non-fatal anomalies (missing dimensions, unusually large
            weights) are logged and recorded in `self.warnings` instead.
        """
        # A score like 1e7 would fail schema validation anyway (it's
        # outside [0, 1]), but the schema's own message ("1e7 is greater
        # than the maximum of 1") doesn't say *why* that's likely wrong.
        # This check exists only to attach that specific explanation
        # before the generic schema error would otherwise win.
        for dim, score in self.scores.items():
            if isinstance(score, (int, float)) and not isinstance(score, bool) and score > SCORE_MAX_ALLOWED:
                raise ValidationError(
                    f"Score for '{dim}' = {score} exceeds the maximum allowed value "
                    f"({SCORE_MAX_ALLOWED}). This usually means a raw percentage was "
                    f"passed instead of a 0-1 value (e.g. 85 instead of 0.85)."
                )

        # The real structural validation. Covers empty dicts (minProperties:
        # 1), unknown dimension names, non-numeric or out-of-range scores,
        # and non-numeric or negative weights. Booleans are rejected too:
        # JSON Schema treats true/false as a distinct type from number,
        # unlike raw Python isinstance(), where bool is a subclass of int.
        try:
            jsonschema.validate(
                instance={"scores": self.scores, "weights": self.weights},
                schema=INPUT_SCHEMA,
            )
        except jsonschema.ValidationError as exc:
            raise ValidationError(f"Input failed schema validation: {exc.message}") from exc

        # Warnings the schema can't express (it can only accept or reject
        # a payload wholesale, not flag part of an otherwise-valid one).
        for dim, weight in self.weights.items():
            if weight > WEIGHT_LARGE_WARN:
                self._warn(f"Weight for '{dim}' = {weight} is unusually large. Verify this is intentional.")

        missing_dims = VALID_DIMENSIONS - set(self.scores)
        if missing_dims:
            self._warn(
                f"Missing trust dimensions: {missing_dims}. These are excluded from "
                "the trust score — consider whether this creates governance blind spots."
            )

        missing_weights = set(self.scores) - set(self.weights)
        if missing_weights:
            self._warn(
                f"No weight provided for dimension(s): {missing_weights}. "
                "Assigning zero weight; these will not affect the trust score."
            )
            for dim in missing_weights:
                self.weights[dim] = 0.0

        logger.info("Input validation passed.")

    def normalize_weights(self) -> dict[str, float]:
        """Normalize weights so they sum to exactly 1.0.

        Frees callers from manually summing weights to 1.0 — a common
        source of error when dimensions are added or removed over time.

        Returns
        -------
        dict[str, float]
            Normalized weights keyed by dimension name.

        Raises
        ------
        NormalizationError
            If all weights are zero (undefined — would divide by zero).
        """
        weight_sum = sum(self.weights.values())
        if abs(weight_sum) < FLOAT_TOLERANCE:
            raise NormalizationError(
                "All weights are zero. Cannot normalize — assign at least one non-zero weight."
            )

        self.normalized_weights = {dim: w / weight_sum for dim, w in self.weights.items()}

        check = sum(self.normalized_weights.values())
        if abs(check - 1.0) > 1e-6:
            raise NormalizationError(f"Normalized weights sum to {check:.8f}, expected 1.0.")

        logger.info("Weights normalised | sum_before=%.4f | sum_after=%.6f", weight_sum, check)
        return self.normalized_weights

    def calculate_score(self) -> TrustScoreResult:
        """Run the full scoring pipeline and return a TrustScoreResult.

        Formula: trust_score = sum(normalized_weight[d] * score[d]) over
        all dimensions present in `scores`. A weighted arithmetic mean was
        chosen over alternatives (geometric mean, sigmoid-curved scoring)
        because stakeholders can understand and challenge a linear formula
        without needing to reason about curve shapes — important for
        audit contexts where the calculation itself may be questioned.

        Returns
        -------
        TrustScoreResult
        """
        self.validate_inputs()
        self.normalize_weights()

        trust_score = sum(
            self.normalized_weights.get(dim, 0.0) * score for dim, score in self.scores.items()
        )
        # Floating-point arithmetic can push the result a hair outside [0, 1].
        trust_score = max(SCORE_MIN, min(SCORE_MAX, trust_score))

        risk_level = self.classify_risk(trust_score)
        risk_flags = self.generate_risk_flags()

        logger.info(
            "Trust score computed | score=%.4f | risk=%s | flags=%s",
            trust_score, risk_level.value, [f.value for f in risk_flags],
        )

        return TrustScoreResult(
            trust_score=trust_score,
            risk_level=risk_level,
            risk_flags=risk_flags,
            scores=dict(self.scores),
            weights=dict(self._raw_weights),
            normalized_weights=dict(self.normalized_weights),
        )

    def classify_risk(self, trust_score: float) -> RiskLevel:
        """Map a trust score to a RiskLevel.

        A score sitting exactly on a boundary is classified into the
        higher-risk tier (e.g. exactly 0.60 -> HIGH, not MEDIUM) — the
        more conservative choice for a governance context.
        """
        if trust_score < RISK_THRESHOLDS["CRITICAL"]:
            return RiskLevel.CRITICAL
        if trust_score <= RISK_THRESHOLDS["HIGH"]:
            return RiskLevel.HIGH
        if trust_score < RISK_THRESHOLDS["MEDIUM"]:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def generate_risk_flags(self) -> list[RiskFlag]:
        """Generate explainable per-dimension risk flags.

        Flags use stricter, per-dimension thresholds (DIMENSION_FLAG_THRESHOLDS)
        rather than the aggregate score thresholds, so a single critical
        failure (e.g. SAFETY at 0.30) is surfaced even when other strong
        dimensions pull the aggregate score up to MEDIUM.
        """
        flags = [
            _DIMENSION_FLAG_MAP[dim]
            for dim, threshold in DIMENSION_FLAG_THRESHOLDS.items()
            if dim in self.scores and self.scores[dim] < threshold
        ]
        for flag in flags:
            logger.warning("Risk flag raised | %s", flag.value)

        if VALID_DIMENSIONS - set(self.scores):
            flags.append(RiskFlag.MISSING_DIMENSION)

        return flags

    def generate_evidence(self, result: TrustScoreResult) -> EvidenceArtifact:
        """Produce a structured, SHA-256 sealed audit evidence artifact.

        Parameters
        ----------
        result : TrustScoreResult
            Output of `calculate_score()`.

        Raises
        ------
        EvidenceError
            If serialization or hashing fails.
        """
        try:
            artifact = EvidenceArtifact(
                artifact_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                evaluator_version=EVALUATOR_VERSION,
                scores=result.scores,
                weights=result.weights,
                normalized_weights={k: round(v, 8) for k, v in result.normalized_weights.items()},
                trust_score=round(result.trust_score, 6),
                risk_level=result.risk_level.value,
                risk_flags=[f.value for f in result.risk_flags],
            )
            artifact.sha256_hash = self.generate_sha256_hash(artifact)
            logger.info(
                "Evidence artifact generated | id=%s | hash=%s...",
                artifact.artifact_id, artifact.sha256_hash[:16],
            )
            return artifact
        except (TypeError, ValueError, OverflowError) as exc:
            logger.error("Evidence artifact generation failed: %s", exc)
            raise EvidenceError(f"Failed to generate evidence artifact: {exc}") from exc

    @staticmethod
    def generate_sha256_hash(artifact: EvidenceArtifact) -> str:
        """Compute the SHA-256 hash of an artifact (hash field excluded from input).

        Why this matters for auditability: any party holding the artifact
        can recompute this hash and compare it to the stored value. A
        mismatch proves the artifact was altered after generation — the
        core integrity guarantee an auditor or regulator relies on.
        SHA-256 (not MD5/SHA-1, both broken for cryptographic use) with
        `sort_keys=True` JSON serialization ensures the same input always
        produces the same hash, regardless of platform or Python version.

        Raises
        ------
        EvidenceError
            If serialization fails.
        """
        try:
            canonical_json = artifact.to_json(include_hash=False)
            return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        except (TypeError, ValueError) as exc:
            logger.error("SHA-256 hashing failed: %s", exc)
            raise EvidenceError(f"SHA-256 hashing failed: {exc}") from exc

    def _warn(self, message: str) -> None:
        """Record a non-fatal warning and log it."""
        logger.warning(message)
        self._warnings.append(message)

    @property
    def warnings(self) -> list[str]:
        """All non-fatal warnings raised during processing, most recent last."""
        return list(self._warnings)


def evaluate(scores: dict[str, float], weights: dict[str, float]) -> EvidenceArtifact:
    """Convenience one-call entry point: score, classify, and seal evidence.

    Equivalent to:
        calc = TrustScoreCalculator(scores, weights)
        result = calc.calculate_score()
        return calc.generate_evidence(result)

    Returns
    -------
    EvidenceArtifact
    """
    calculator = TrustScoreCalculator(scores, weights)
    result = calculator.calculate_score()
    return calculator.generate_evidence(result)
