# Trust Score Calculator

When you're deploying an AI system, someone will eventually ask: *"How do we know we can trust this?"* This tool gives you a defensible, auditable answer.

`trust.py` scores an AI system across six governance dimensions, produces a single trust score, and generates a sealed audit artifact — ready for compliance review, risk reporting, or stakeholder sign-off.

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
| LOW | ≥ 0.75 |
| MEDIUM | > 0.60 and < 0.75 |
| HIGH | 0.40 – 0.60 |
| CRITICAL | < 0.40 |

> The boundary at `0.60` deliberately falls into HIGH — when in doubt, the tool errs on the side of caution.

## Why it matters

AI governance reviews are only as good as the evidence behind them. This tool makes sure every run produces the same result, every input is validated, and every output is traceable. Less scrambling at audit time. More confidence in what you're signing off on.

## Honest limitations

This tool scores what you give it — it doesn't collect data on its own. The six dimensions are fixed in this version, and it's designed for point-in-time snapshots rather than ongoing monitoring. Flagged risks still need a human to interpret and act on them.

## How to use it

Run the sample evaluation:
```bash
python -m pip install jsonschema
python trust.py
python -m unittest trust.py
```

Or import it directly into your pipeline:
```python
from trust import evaluate
result = evaluate(scores={...}, weights={...})
```

It fits into any Python workflow, CI/CD step, or governance review process without much effort.

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
  "sha256_hash": "1e55bf5959636123aced6222078c0aaa7759585adf1b8fce1a7064a784783c35",
  "timestamp": "2026-06-25T04:55:32.083462+00:00"
}
```

In this example, fairness is the weak spot — everything else is solid, but that one flag is enough to land the system in MEDIUM risk territory.

## What gets validated

- Scores must be between `0.0` and `1.0`
- Weights must be non-negative and are auto-normalized before scoring
- Only the six supported dimensions are accepted
- Missing dimensions raise a warning, not a crash

## Test coverage

The suite covers input validation, weight normalization, risk classification boundaries, end-to-end scoring, artifact structure, and hash integrity.

## What's next

Planned improvements include custom dimension support, configurable risk thresholds, and batch evaluation across multiple systems.

## Requirements

- Python 3.9+
- `jsonschema` (the only external dependency)



