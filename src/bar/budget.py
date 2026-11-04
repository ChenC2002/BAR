"""Fixed task-specific empirical CDFs for Eq. (4)."""

from bisect import bisect_right

from pydantic import Field, model_validator

from bar.config import ReasoningConfig
from bar.schemas import Probability, Record


class BudgetAllocator(Record):
    uncertainties: dict[str, list[Probability]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sorted_samples(self) -> "BudgetAllocator":
        if not self.uncertainties or any(not xs for xs in self.uncertainties.values()):
            raise ValueError("CDFs require training predictions for every task")
        self.uncertainties = {task: sorted(xs) for task, xs in self.uncertainties.items()}
        return self

    @classmethod
    def fit(cls, training_predictions: dict[str, list[float]]) -> "BudgetAllocator":
        for values in training_predictions.values():
            if any(not 0 <= p <= 1 for p in values):
                raise ValueError("probabilities must be finite and in [0, 1]")
        return cls(
            uncertainties={
                task: [1 - abs(2 * p - 1) for p in values]
                for task, values in training_predictions.items()
            }
        )

    def allocate(self, task: str, probability: float, config: ReasoningConfig) -> float:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be finite and in [0, 1]")
        if task not in self.uncertainties:
            raise ValueError(f"no training uncertainty CDF for {task}")
        reference = self.uncertainties[task]
        percentile = bisect_right(reference, 1 - abs(2 * probability - 1)) / len(reference)
        return config.min_budget + (config.max_budget - config.min_budget) * percentile
