"""REINFORCE helper used in Stage 3 policy optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunningMeanBaseline:
    decay: float = 0.9
    value: float = 0.0
    initialized: bool = False

    def advantage(self, reward: float) -> float:
        if not self.initialized:
            self.value = reward
            self.initialized = True
            return 0.0
        advantage = reward - self.value
        self.value = self.decay * self.value + (1.0 - self.decay) * reward
        return advantage


def reinforce_loss(log_probability_sum: float, reward: float, baseline: RunningMeanBaseline) -> float:
    """Return the scalar policy-gradient loss -A * sum log pi(a_t)."""

    advantage = baseline.advantage(reward)
    return -advantage * log_probability_sum
