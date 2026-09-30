from collections.abc import Callable
from typing import Literal

from .phase_metric_base import MetricHandlerBase as NullMetrics
from .phase_collector_base import CollectorBase as NullCollector

ValidationOwner = Literal["trainer", "handlers"]


class TrainingModule:

    def __init__(
            self,
            trainer,
            params,
            *,
            metrics=None,
            collector=None,
            validation_owner: ValidationOwner = "handlers",
            checkpoint_handler: Callable | None = None,
    ):

        self.trainer = trainer
        self.params = params

        if validation_owner not in {"trainer", "handlers"}:
            raise ValueError(
                "validation_owner must be 'trainer' or 'handlers'"
            )

        if validation_owner == "trainer":
            required = ("reset_metrics", "update_metrics", "report_metrics")
            missing = [
                name
                for name in required
                if not callable(getattr(trainer, name, None))
            ]
            if missing:
                raise TypeError(
                    "trainer-owned validation requires methods: "
                    + ", ".join(missing)
                )

        self.validation_owner = validation_owner
        self.metrics = metrics if metrics is not None else NullMetrics()
        self.collector = collector if collector is not None else NullCollector()
        self.checkpoint_handler = checkpoint_handler
        self._step = 0

    @property
    def current_step(self):
        return self._step

    @property
    def model(self):
        return self.trainer.model

    def compute(self, batch ):
        return self.trainer.compute(batch, self.params)

    def train_context(self):
        return self.trainer.train_context()

    def eval_context(self):
        return self.trainer.eval_context()

    # ----- lifecycle -----

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_step_begin(self, step):
        self._step = step
        self.params.step(step)

    def on_train_payload(self, payload):
        pass

    def on_checkpoint(
            self,
            optimizer,
            scheduler = None
    ):
        if self.checkpoint_handler is not None:
            self.checkpoint_handler(
                self,
                optimizer,
                scheduler,
            )

    #---------------------------------------------
    def on_log(self, payload):

        if payload is None:
            return None

        if hasattr(payload, "report"):
            return payload.report()

        if hasattr(payload, "summarize"):
            return payload.summarize()

        return payload

    #---------------------------------------------
    def on_validation_begin(self,  step):

        if self.validation_owner == "trainer":
            self.trainer.reset_metrics()
            return

        self.metrics.begin(self.params)

        self.collector.begin(
            self.params,
            step
        )

    #---------------------------------------------
    def on_validation_payload(self, payload):

        if self.validation_owner == "trainer":
            self.trainer.update_metrics(payload)
            return

        self.metrics.update(payload)
        self.collector.update(payload)

    #---------------------------------------------
    def on_validation_end(self):

        if self.validation_owner == "trainer":
            return self.trainer.report_metrics()

        summary = self.metrics.summarize()

        if hasattr(self.metrics, "report"):
            summary = self.metrics.report(summary)

        self.collector.finalize()

        return summary
