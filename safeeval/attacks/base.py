"""Base class for attack/red-team datasets."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Iterator


@dataclass
class AttackSample:
    id: str
    prompt: str
    category: str
    source: str
    expected_refusal: bool = True
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAttackSet(ABC):
    def __init__(self, num_samples: int = 50):
        self.num_samples = num_samples
        self._samples: List[AttackSample] = []

    @abstractmethod
    def load(self) -> List[AttackSample]:
        pass

    def __iter__(self) -> Iterator[AttackSample]:
        if not self._samples:
            self._samples = self.load()
        yield from self._samples

    def __len__(self) -> int:
        if not self._samples:
            self._samples = self.load()
        return len(self._samples)
