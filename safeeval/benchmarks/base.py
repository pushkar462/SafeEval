"""Base class for capability benchmarks."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Iterator, Optional


@dataclass
class BenchmarkSample:
    id: str
    prompt: str
    reference: Optional[str] = None
    choices: List[str] = field(default_factory=list)
    correct_choice: Optional[int] = None
    category: str = "general"
    source: str = ""
    metadata: dict = field(default_factory=dict)


class BaseBenchmark(ABC):
    def __init__(self, num_samples: int = 100, split: str = "validation"):
        self.num_samples = num_samples
        self.split = split
        self._samples: List[BenchmarkSample] = []

    @abstractmethod
    def load(self) -> List[BenchmarkSample]:
        pass

    def __iter__(self) -> Iterator[BenchmarkSample]:
        if not self._samples:
            self._samples = self.load()
        yield from self._samples

    def __len__(self) -> int:
        if not self._samples:
            self._samples = self.load()
        return len(self._samples)
