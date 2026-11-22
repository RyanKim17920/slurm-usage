"""Usage over time in linear time, however long each allocation is."""

from typing import List

from .select import effective


class Series:
    """Exact integer usage per bin. Full bins are kept as a difference array of rates."""

    __slots__ = ("edges", "partial", "change")

    def __init__(self, edges: List[int]) -> None:
        self.edges = edges
        self.partial = [0] * (len(edges) - 1)
        self.change = [0] * len(edges)

    def add(self, i: int, j: int, p: int, q: int, rate: int) -> None:
        """``rate`` over [p, q), where p is in bin i and q in bin j."""
        if i == j:
            self.partial[i] += rate * (q - p)
            return
        edges = self.edges
        self.partial[i] += rate * (edges[i + 1] - p)
        self.partial[j] += rate * (q - edges[j])
        self.change[i + 1] += rate
        self.change[j] -= rate

    def add_suspended(self, i: int, j: int, p: int, q: int, rate: int, begun: int,
                      duration: int, suspended: int) -> None:
        edges, before = self.edges, effective(p - begun, duration, suspended)
        for at in range(i, j + 1):
            after = effective(min(q, edges[at + 1]) - begun, duration, suspended)
            self.partial[at] += rate * (after - before)
            before = after

    def values(self) -> List[int]:
        edges, rate, out = self.edges, 0, []
        for at, value in enumerate(self.partial):
            rate += self.change[at]
            out.append(value + rate * (edges[at + 1] - edges[at]))
        return out
