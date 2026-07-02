"""Enums and data containers for the trust scoring engine."""

import json
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(str, Enum):
    """Four-tier risk classification modelled after enterprise risk frameworks.

    CRITICAL (< 0.40)
        Trust too low for deployment. A High-Risk AI system scoring here
        would likely fail conformity assessment under frameworks such as
        the EU AI Act. Requires remediation before any deployment.

    HIGH (0.40-0.60)
        Significant governance gaps. Deployable only in controlled,
        monitored environments with compensating controls.

    MEDIUM (0.60-0.75)
        Acceptable for limited deployment with documented risk acceptance
        and ongoing monitoring. Where most real-world systems land on
        first evaluation.

    LOW (> 0.75)
        Strong trust posture, meets enterprise deployment standards.
        Does not mean zero risk.
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskFlag(str, Enum):
    """Explainable per-dimension risk flags surfaced in the evidence artifact."""

    ACCURACY_RISK = "ACCURACY_RISK"
    ROBUSTNESS_RISK = "ROBUSTNESS_RISK"
    FAIRNESS_RISK = "FAIRNESS_RISK"
    SAFETY_RISK = "SAFETY_RISK"
    PRIVACY_RISK = "PRIVACY_RISK"
    TRANSPARENCY_RISK = "TRANSPARENCY_RISK"
    MISSING_DIMENSION = "MISSING_DIMENSION"


@dataclass
class TrustScoreResult:
    """Immutable result of a single scoring run.

    Separating the result from the calculator lets callers pass it around
    (e.g. into `generate_evidence`) without coupling to calculator state.
    """

    trust_score: float
    risk_level: RiskLevel
    risk_flags: list[RiskFlag]
    scores: dict[str, float]
    weights: dict[str, float]
    normalized_weights: dict[str, float]


@dataclass
class EvidenceArtifact:
    """Structured, tamper-evident audit artifact.

    `sha256_hash` lets any downstream party — auditor, regulator, internal
    compliance system — verify the artifact hasn't been altered since it
    was generated. A lightweight substitute for a digital signature where
    full PKI infrastructure isn't justified.
    """

    artifact_id: str
    timestamp: str
    evaluator_version: str
    scores: dict[str, float]
    weights: dict[str, float]
    normalized_weights: dict[str, float]
    trust_score: float
    risk_level: str
    risk_flags: list[str]
    sha256_hash: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (hash excluded — used as the pre-hash payload)."""
        return {
            "artifact_id": self.artifact_id,
            "timestamp": self.timestamp,
            "evaluator_version": self.evaluator_version,
            "scores": self.scores,
            "weights": self.weights,
            "normalized_weights": self.normalized_weights,
            "trust_score": round(self.trust_score, 6),
            "risk_level": self.risk_level,
            "risk_flags": self.risk_flags,
        }

    def to_json(self, include_hash: bool = True) -> str:
        """Serialize to canonical JSON.

        `sort_keys=True` guarantees a deterministic byte string regardless
        of dict insertion order or Python version — required for the hash
        in `sha256_hash` to be reproducible by anyone re-verifying it.
        """
        data = self.to_dict()
        if include_hash:
            data["sha256_hash"] = self.sha256_hash
        return json.dumps(data, sort_keys=True, indent=2)
