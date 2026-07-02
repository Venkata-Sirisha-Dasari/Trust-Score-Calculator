"""Sample evaluation run: an AI system used for loan-approval decisions.

Run directly with: python -m src.demo
"""

import logging

from .calculator import TrustScoreCalculator
from .constants import VALID_DIMENSIONS

# Configuring logging here (not inside the library) means importing the
# calculator never has side effects on a host application's logging setup.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)


def run_sample_evaluation():
    """Evaluate a loan-approval AI system (EU AI Act Annex III High-Risk).

    Safety and fairness are weighted highest here because a biased or
    unsafe loan decision directly affects someone's livelihood.
    """
    scores = {
        "accuracy": 0.88,
        "robustness": 0.72,
        "fairness": 0.51,
        "safety": 0.65,
        "privacy": 0.80,
        "transparency": 0.58,
    }
    weights = {
        "accuracy": 0.20,
        "robustness": 0.15,
        "fairness": 0.25,
        "safety": 0.25,
        "privacy": 0.10,
        "transparency": 0.05,
    }

    print("\n" + "=" * 68)
    print("  TRUST SCORE EVALUATION ENGINE — Sample Run")
    print("  Use Case: Loan Approval AI (EU AI Act High-Risk)")
    print("=" * 68)

    calc = TrustScoreCalculator(scores, weights)
    result = calc.calculate_score()

    print(f"\n  {'Dimension':<16} {'Score':>8}  {'Weight (norm)':>14}")
    print("  " + "-" * 44)
    for dim in sorted(VALID_DIMENSIONS):
        s = result.scores.get(dim, float("nan"))
        w = result.normalized_weights.get(dim, 0.0)
        print(f"  {dim:<16} {s:>8.4f}  {w:>14.4f}")

    print(f"\n  Trust Score  : {result.trust_score:.4f}")
    print(f"  Risk Level   : {result.risk_level.value}")
    print(f"  Risk Flags   : {[f.value for f in result.risk_flags] or 'None'}")

    if calc.warnings:
        print("\n  Warnings:")
        for w in calc.warnings:
            print(f"    - {w}")

    artifact = calc.generate_evidence(result)
    print("\n" + "=" * 68)
    print("  EVIDENCE ARTIFACT (SHA-256 Sealed)")
    print("=" * 68)
    print(artifact.to_json())
    return artifact


if __name__ == "__main__":
    run_sample_evaluation()
