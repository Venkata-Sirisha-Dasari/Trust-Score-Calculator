"""Domain-specific exceptions.

Raising these instead of raw ValueError/TypeError lets callers catch
scoring-specific failures without accidentally swallowing unrelated bugs.
"""


class TrustScoreError(Exception):
    """Base exception for all trust scoring errors."""


class ValidationError(TrustScoreError):
    """Raised when input scores or weights fail validation."""


class NormalizationError(TrustScoreError):
    """Raised when weight normalization is impossible (e.g. all weights zero)."""


class EvidenceError(TrustScoreError):
    """Raised when evidence artifact generation or hashing fails."""
