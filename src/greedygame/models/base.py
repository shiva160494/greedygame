from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """All models return a partial probability distribution over outcomes."""
    name: str

    @abstractmethod
    def predict_proba(self, history: list[str]) -> dict[str, float]:
        raise NotImplementedError
