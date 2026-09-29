from __future__ import annotations

from abc import ABC, abstractmethod

from .training_context import TrainingContext
from ..param_wrapper import ParameterWrapper

class TrainingObjective(ABC):
    """Minimal contract for one reusable horizontal slice of training."""

    name: str

    @abstractmethod
    def compute(
        self,
        context: TrainingContext,
        params: ParameterWrapper, #Any = None,
    ):
        raise NotImplementedError

    # payload reporting
    @abstractmethod
    def report(self):
        raise NotImplementedError

    #------------------------------------
    #  metric handling functionality
    #____________________________________
    @abstractmethod
    def reset_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def update_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def report_metrics(self):
        raise NotImplementedError