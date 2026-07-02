# Trust Score Calculator

When you're deploying an AI system, someone will eventually ask: *"How do we know we can trust this?"* This tool gives you a defensible, auditable answer.

It scores an AI system across six governance dimensions, produces a single trust score, and generates a SHA-256 sealed audit artifact — ready for compliance review, risk reporting, or stakeholder sign-off.

## What it evaluates

Six dimensions, each scored between `0.0` and `1.0`:

- **Accuracy** — does it get things right?
- **Robustness** — does it hold up under pressure?
- **Fairness** — does it treat everyone equally?
- **Safety** — does it avoid harmful outcomes?
- **Privacy** — does it respect sensitive data?
- **Transparency** — can you see how it works?

## What you get back

- A single trust score between `0.0` and `1.0`
- A risk classification: `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW`
- Flags pointing to whichever dimensions need attention
- A SHA-256 sealed JSON artifact you can hand to an auditor

## Risk levels explained

| Risk Level | Trust Score |
|------------|-------------|
| LOW | > 0.75 |
| MEDIUM | 0.60 < score < 0.75 |
| HIGH | 0.40 ≤ score ≤ 0.60 |
| CRITICAL | < 0.40 |

> A score sitting exactly on a boundary (e.g. exactly `0.60`) is classified into the higher-risk tier — the more conservative choice for a governance context.

## Project structure

```text
trust-score-calculator/
│
├── data/
│   └── sample_input.json      # example scores/weights payload
│
├── schema/
│   └── trust_schema.json      # single source of truth for valid dimensions & bounds
│
├── src/
│   ├── __init__.py            # public API (TrustScoreCalculator, evaluate)
│   ├── constants.py           # dimensions, bounds, risk thresholds
│   ├── exceptions.py          # ValidationError, NormalizationError, EvidenceError
│   ├── models.py              # RiskLevel, RiskFlag, TrustScoreResult, EvidenceArtifact
│   ├── calculator.py          # TrustScoreCalculator engine
│   └── demo.py                # runnable sample evaluation
│
├── tests/
│   └── test_calculator.py     # 21 unit tests
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

```bash
git clone <repo-url>
cd trust-score-calculator
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Running the application

Run the sample evaluation (loan-approval AI scenario):

```bash
python -m src.demo
```

Or use it in your own code:

```python
from src import evaluate

artifact = evaluate(
    scores={"accuracy": 0.88, "fairness": 0.51, "safety": 0.65},
    weights={"accuracy": 0.4, "fairness": 0.3, "safety": 0.3},
)
print(artifact.to_json())
```

For more control (inspecting warnings, intermediate normalized weights, etc.), use the class directly:

```python
from src import TrustScoreCalculator

calc = TrustScoreCalculator(scores, weights)
result = calc.calculate_score()
artifact = calc.generate_evidence(result)

print(calc.warnings)          # any non-fatal issues found during validation
print(result.risk_level)      # RiskLevel.MEDIUM
print(artifact.sha256_hash)   # 64-character hex digest
```

## Running tests

```bash
python -m unittest tests.test_calculator -v
```

All 21 tests should pass. Coverage includes input validation, weight normalization, risk classification boundaries (including exact threshold edges), end-to-end scoring, evidence artifact structure, and SHA-256 hash determinism.

## What the output looks like

```json
{
  "artifact_id": "7d07b2e2-549c-4727-ab62-c0f378185d37",
  "evaluator_version": "1.0.0",
  "trust_score": 0.683,
  "risk_level": "MEDIUM",
  "risk_flags": ["FAIRNESS_RISK"],
  "scores": {
    "accuracy": 0.88,
    "fairness": 0.51,
    "privacy": 0.80,
    "robustness": 0.72,
    "safety": 0.65,
    "transparency": 0.58
  },
  "sha256_hash": "de7c561f36ecc213ac98bd888a64c71369754c35f3aaf4b3d59fa4e35673c36d",
  "timestamp": "2026-07-02T06:16:32.686831+00:00"
}
```

Fairness is the weak spot here — everything else is solid, but that one flag is enough to land the system in MEDIUM risk territory.

## What gets validated

Validation happens in two layers, each with a distinct job:

1. **`schema/trust_schema.json`** — the actual structural validator, enforced via `jsonschema.validate()`. It's the single source of truth for: which six dimensions are allowed, that scores are numeric in `[0.0, 1.0]`, that weights are numeric and non-negative, and that at least one dimension is present. `VALID_DIMENSIONS` in `src/constants.py` is *derived* from this file at import time, so the schema and the Python code can never drift out of sync. Booleans are rejected here too — JSON Schema treats `true`/`false` as a distinct type from `number`, unlike raw Python `isinstance()`, where `bool` is a subclass of `int`.
2. **A small amount of manual code** handles only what a JSON Schema can't express: warnings rather than hard failures (missing dimensions, missing weights, unusually large weights — all available via `calc.warnings`), and one specific, more actionable error message for a likely data-entry mistake (e.g. `85` passed instead of `0.85`) ahead of the generic schema error.

Nothing in layer 2 duplicates a check the schema already performs.

## Design notes

- **Weighted arithmetic mean, not geometric mean or a sigmoid curve.** A linear formula is something a stakeholder can verify by hand and challenge in an audit — important when the calculation itself may be questioned.
- **Per-dimension flag thresholds are stricter than the aggregate score thresholds.** A single failed dimension (e.g. Safety at 0.30) is a categorical risk even if other dimensions pull the aggregate score up to MEDIUM.
- **JSON Schema as the single source of truth, not decoration.** An earlier draft ran a JSON Schema check *and* a full set of manual checks that duplicated it — and because the schema ran first, it silently swallowed the more specific manual error messages before they could ever fire. The current design uses the schema (`schema/trust_schema.json`) as the actual validator, derives `VALID_DIMENSIONS` from it so the two can't drift apart, and keeps only the handful of manual checks the schema genuinely can't express (warnings, and one more actionable error message for a common data-entry mistake).

## Honest limitations

This tool scores what you give it — it doesn't collect data on its own. The six dimensions are fixed in this version, and it's designed for point-in-time snapshots rather than ongoing monitoring. Flagged risks still need a human to interpret and act on them.

## Future improvements

- Custom/extensible dimension support
- Configurable risk thresholds per deployment context
- Batch evaluation across multiple systems
- Optional CLI (`python -m src.demo --input data/sample_input.json`)

## Requirements

- Python 3.9+ (uses built-in generic types like `dict[str, float]`)
- `jsonschema>=4.17` (the only runtime dependency — see `requirements.txt`)
