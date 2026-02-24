"""Data models for variable scoring."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VariableScore:
    """Multi-dimensional score for a variable's usefulness for code discovery."""

    config_value: int = 1
    architectural_role: int = 1
    data_definition: int = 1
    general_usefulness: int = 1

    @property
    def max_score(self) -> int:
        """Return the highest dimension score."""
        return max(
            self.config_value,
            self.architectural_role,
            self.data_definition,
            self.general_usefulness,
        )

    def passes_threshold(self, threshold: int = 5) -> bool:
        """Check if any dimension scores at or above threshold."""
        return self.max_score >= threshold

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return scores as a tuple for compact display."""
        return (
            self.config_value,
            self.architectural_role,
            self.data_definition,
            self.general_usefulness,
        )


@dataclass
class VariableScoringConfig:
    """Configuration for variable scoring."""

    threshold: int = 5
    temperature: float = 0.1
    token_budget: int = 800
    max_retries: int = 2


@dataclass
class ScoringResult:
    """Result of the variable scoring phase."""

    total_variables: int = 0
    kept: int = 0
    dropped: int = 0
    batch_count: int = 0
    elapsed: float = 0.0
    errors: int = 0
    scores: dict[str, VariableScore] = field(default_factory=dict)
