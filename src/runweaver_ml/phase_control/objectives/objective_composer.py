from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from torch import Tensor


from .training_context import TrainingContext
from .training_objective import TrainingObjective

from ..execution.training_model_wrapper_base import system_state

from ..execution.phase_trainer_base import TrainerBase

ForwardStrategy = Callable[[TrainingContext], TrainingContext]


class ObjectiveComposer(TrainerBase):
    """Executes one forward strategy and multiple objectives for a batch."""

    def __init__(
        self,
        model,
        forward, #: ForwardStrategy,
        objectives: Iterable[TrainingObjective],
        train_mode,
        eval_mode
    ):
        super().__init__(
            model, train_mode, eval_mode
        )

        self.forward = forward
        self.objectives = objectives
        self.last_total_loss = 0.0

        if not self.objectives:
            raise ValueError(
                "ObjectiveComposer requires at least one objective"
            )

        names = [objective.name for objective in self.objectives]

        if len(names) != len(set(names)):
            raise ValueError(
                f"Duplicate objective names: {names}"
            )


    def compute( self, batch, params ):

        context = self.forward(
            TrainingContext(batch=batch),
            self.model
        )

        self.last_total_loss = 0.0

        for objective in self.objectives:
            loss = objective.compute(
                context,
                params,
            )
            self.last_total_loss += loss


        return  self.last_total_loss, self

    def report(self):

        reports = [f"loss={self.last_total_loss:.6f}"]

        for objective in self.objectives:
            report = objective.report()

            if report:
                reports.append(report)

        return " | ".join(reports)


    def reset_metrics(self):

        for objective in self.objectives:
            objective.reset_metrics()


    def update_metrics(self, payload):

        for objective in self.objectives:
            objective.update_metrics(payload)


    def report_metrics(self):

        reports = []

        for objective in self.objectives:
            report = objective.report_metrics()

            if report:
                reports.append(report)

        return "\n".join(reports)