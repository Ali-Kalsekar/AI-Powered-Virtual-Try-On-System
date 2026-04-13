from __future__ import annotations

from time import perf_counter


class FPSCounter:
    def __init__(self, avg_window: int = 20) -> None:
        self.avg_window = max(1, avg_window)
        self.prev_time = perf_counter()
        self.samples: list[float] = []

    def update(self) -> float:
        current = perf_counter()
        dt = max(1e-6, current - self.prev_time)
        self.prev_time = current

        fps = 1.0 / dt
        self.samples.append(fps)
        if len(self.samples) > self.avg_window:
            self.samples.pop(0)

        return sum(self.samples) / len(self.samples)
