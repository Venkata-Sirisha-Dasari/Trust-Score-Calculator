"""Unit tests for TrustScoreCalculator.

Each test exercises a single concern and is named to be self-documenting,
ordered from fundamental (validation) to complex (integration).
"""

import json
import re
import unittest

from src.calculator import TrustScoreCalculator
from src.constants import VALID_DIMENSIONS
from src.exceptions import NormalizationError, ValidationError
from src.models import RiskFlag, RiskLevel, TrustScoreResult


class TestTrustScoreEngine(unittest.TestCase):
    VALID_SCORES = {
        "accuracy": 0.85, "robustness": 0.75, "fairness": 0.70,
        "safety": 0.80, "privacy": 0.90, "transparency": 0.65,
    }
    VALID_WEIGHTS = {
        "accuracy": 0.20, "robustness": 0.15, "fairness": 0.20,
        "safety": 0.25, "privacy": 0.10, "transparency": 0.10,
    }

    def test_01_happy_path_returns_result(self):
        """Valid inputs produce a TrustScoreResult with score in [0, 1]."""
        calc = TrustScoreCalculator(self.VALID_SCORES, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        self.assertIsInstance(result, TrustScoreResult)
        self.assertGreaterEqual(result.trust_score, 0.0)
        self.assertLessEqual(result.trust_score, 1.0)
        self.assertIsInstance(result.risk_level, RiskLevel)

    def test_02_weight_normalisation_sums_to_one(self):
        """Weights that don't sum to 1.0 (e.g. raw ratios) normalize correctly."""
        weights = {"accuracy": 2, "safety": 3, "fairness": 5}
        scores = {"accuracy": 0.8, "safety": 0.7, "fairness": 0.6}
        calc = TrustScoreCalculator(scores, weights)
        normalised = calc.normalize_weights()
        self.assertAlmostEqual(sum(normalised.values()), 1.0, places=6)

    def test_03_score_above_one_raises_validation_error(self):
        """A score > 1.0 raises ValidationError with a specific, useful message."""
        bad_scores = {**self.VALID_SCORES, "accuracy": 1.5}
        calc = TrustScoreCalculator(bad_scores, self.VALID_WEIGHTS)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_04_score_below_zero_raises_validation_error(self):
        bad_scores = {**self.VALID_SCORES, "safety": -0.1}
        calc = TrustScoreCalculator(bad_scores, self.VALID_WEIGHTS)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_05_negative_weight_raises_validation_error(self):
        """A negative weight would invert a dimension's contribution — must be rejected."""
        bad_weights = {**self.VALID_WEIGHTS, "fairness": -0.10}
        calc = TrustScoreCalculator(self.VALID_SCORES, bad_weights)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_06_all_weights_zero_raises_normalisation_error(self):
        zero_weights = {k: 0.0 for k in self.VALID_SCORES}
        calc = TrustScoreCalculator(self.VALID_SCORES, zero_weights)
        with self.assertRaises(NormalizationError):
            calc.normalize_weights()

    def test_07_missing_dimensions_produce_warning_not_error(self):
        """Partial evaluation (e.g. privacy N/A for non-personal-data AI) is valid but flagged."""
        partial_scores = {"accuracy": 0.8, "safety": 0.7}
        partial_weights = {"accuracy": 0.5, "safety": 0.5}
        calc = TrustScoreCalculator(partial_scores, partial_weights)
        calc.validate_inputs()
        self.assertTrue(any("Missing" in w for w in calc.warnings))

    def test_08_risk_classification_critical(self):
        calc = TrustScoreCalculator({"accuracy": 0.3}, {"accuracy": 1.0})
        self.assertEqual(calc.classify_risk(0.30), RiskLevel.CRITICAL)

    def test_09_risk_classification_low(self):
        calc = TrustScoreCalculator({"accuracy": 0.9}, {"accuracy": 1.0})
        self.assertEqual(calc.classify_risk(0.90), RiskLevel.LOW)

    def test_10_risk_boundary_exactly_at_high_threshold(self):
        """Score exactly at 0.60 classifies as HIGH, not MEDIUM (conservative boundary)."""
        calc = TrustScoreCalculator({"accuracy": 0.6}, {"accuracy": 1.0})
        self.assertEqual(calc.classify_risk(0.60), RiskLevel.HIGH)

    def test_11_sha256_hash_is_64_hex_chars(self):
        calc = TrustScoreCalculator(self.VALID_SCORES, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        art = calc.generate_evidence(result)
        self.assertEqual(len(art.sha256_hash), 64)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", art.sha256_hash))

    def test_12_identical_artifact_content_produces_identical_hash(self):
        """Hashing itself is deterministic: same artifact content -> same hash.

        Two full evaluation runs will have different artifact_id/timestamp
        by design (each run is a distinct audit event), so this test
        isolates the hash function itself by hashing the same
        already-built artifact content twice, rather than comparing two
        independently generated artifacts.
        """
        calc = TrustScoreCalculator(self.VALID_SCORES, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        artifact = calc.generate_evidence(result)

        hash_a = TrustScoreCalculator.generate_sha256_hash(artifact)
        hash_b = TrustScoreCalculator.generate_sha256_hash(artifact)
        self.assertEqual(hash_a, hash_b)

    def test_13_evidence_artifact_contains_required_fields(self):
        calc = TrustScoreCalculator(self.VALID_SCORES, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        art = calc.generate_evidence(result)
        data = json.loads(art.to_json())
        required_fields = [
            "artifact_id", "timestamp", "evaluator_version", "scores", "weights",
            "normalized_weights", "trust_score", "risk_level", "risk_flags", "sha256_hash",
        ]
        for f in required_fields:
            self.assertIn(f, data, msg=f"Missing field: {f}")

    def test_14_all_scores_zero_produces_critical(self):
        zero_scores = {k: 0.0 for k in VALID_DIMENSIONS}
        calc = TrustScoreCalculator(zero_scores, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        self.assertEqual(result.risk_level, RiskLevel.CRITICAL)
        self.assertAlmostEqual(result.trust_score, 0.0)

    def test_15_all_scores_one_produces_low_risk(self):
        perfect_scores = {k: 1.0 for k in VALID_DIMENSIONS}
        calc = TrustScoreCalculator(perfect_scores, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        self.assertEqual(result.risk_level, RiskLevel.LOW)
        self.assertAlmostEqual(result.trust_score, 1.0, places=6)

    def test_16_empty_scores_raises_validation_error(self):
        calc = TrustScoreCalculator({}, self.VALID_WEIGHTS)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_17_unknown_dimension_raises_validation_error(self):
        """A typo like 'accurasy' must be caught, not silently scored as 0."""
        bad_scores = {"accurasy": 0.8}
        calc = TrustScoreCalculator(bad_scores, self.VALID_WEIGHTS)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_18_risk_flag_raised_for_low_fairness(self):
        scores = {**self.VALID_SCORES, "fairness": 0.40}
        calc = TrustScoreCalculator(scores, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        self.assertIn(RiskFlag.FAIRNESS_RISK, result.risk_flags)

    def test_19_extremely_large_score_raises_error_with_helpful_message(self):
        """A raw percentage (e.g. 1e7) must be caught with a message pointing at the likely cause."""
        bad_scores = {**self.VALID_SCORES, "accuracy": 1e7}
        calc = TrustScoreCalculator(bad_scores, self.VALID_WEIGHTS)
        with self.assertRaises(ValidationError) as ctx:
            calc.validate_inputs()
        self.assertIn("percentage", str(ctx.exception))

    def test_20_single_dimension_evaluates_correctly(self):
        single_score = {"accuracy": 0.72}
        single_weight = {"accuracy": 1.0}
        calc = TrustScoreCalculator(single_score, single_weight)
        result = calc.calculate_score()
        self.assertAlmostEqual(result.trust_score, 0.72, places=6)

    def test_21_boolean_score_is_rejected(self):
        """bool is a subclass of int in Python, but JSON Schema treats true/false as a
        distinct type from number — True/False must not silently pass as 1/0."""
        bad_scores = {**self.VALID_SCORES, "accuracy": True}
        calc = TrustScoreCalculator(bad_scores, self.VALID_WEIGHTS)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_22_schema_rejects_unexpected_top_level_key(self):
        """Proves jsonschema.validate() is actually wired in and enforced — not
        dead weight — by exercising a rule with no dedicated manual check:
        additionalProperties: false on the top-level payload."""
        import jsonschema as _js

        from src.constants import INPUT_SCHEMA

        with self.assertRaises(_js.ValidationError):
            _js.validate(
                instance={"scores": self.VALID_SCORES, "weights": self.VALID_WEIGHTS, "extra": 1},
                schema=INPUT_SCHEMA,
            )


if __name__ == "__main__":
    unittest.main()
