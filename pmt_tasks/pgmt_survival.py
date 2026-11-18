"""Recovery probability driven by a rolling window of completed ordinary episodes."""
from collections import deque
import math


class NormalEpisodeSurvival:
    window = 100
    review_interval = 20

    def __init__(self):
        self.recent = deque(maxlen=self.window)
        self.total = 0
        self.target = self.probability = .1

    @property
    def mean(self):
        return sum(self.recent) / len(self.recent) if self.recent else 0.

    def observe(self, ratios):
        # Consume simultaneous completions in order. A large reset can cross
        # several review boundaries, each with its own last-100 window.
        for value in ratios:
            self.recent.append(value)
            self.total += 1
            if self.total % self.review_interval == 0:
                self.target = .1 + .4 * self.mean
                self.probability += max(-.02, min(.02, self.target - self.probability))

    def state_dict(self):
        return dict(total=self.total, recent=list(self.recent), target=self.target, probability=self.probability)

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict):
            raise ValueError("Missing ordinary episode survival history")
        total, recent = state.get("total"), state.get("recent")
        if (type(total) is not int or total < 0 or not isinstance(recent, list)
                or len(recent) != min(cls.window, total)
                or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in recent)):
            raise ValueError("Invalid ordinary episode survival history")
        values = [state.get(k) for k in ("target", "probability")]
        if any(type(x) not in (int, float) or not .1 <= x <= .5 for x in values):
            raise ValueError("Invalid terrain curriculum probability")
        if total < cls.review_interval and values != [.1, .1]:
            raise ValueError("Terrain curriculum probability changed before its first review")
        result = cls()
        result.total = total
        result.recent.extend(recent)
        result.target, result.probability = values
        return result
