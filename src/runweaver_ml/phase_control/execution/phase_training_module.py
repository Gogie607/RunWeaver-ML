from abc import ABC, abstractmethod
from .phase_metric_base import MetricHandlerBase as NullMetrics
from .phase_collector_base import CollectorBase as NullCollector

class TrainingModule(ABC):

    def __init__(
            self,
            trainer,
            params,
            *,
            metrics=NullMetrics(),
            collector=NullCollector()
    ):

        self.trainer = trainer
        self.params = params

        self.metrics = metrics
        self.collector = collector
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
        ckpt = self.params.get("checkpoint")
        ckpt_dir = ckpt.get("dir")
        if ckpt_dir is not None:
            self.trainer.model.save_checkpoint(
                ckpt_dir,
                self.current_step,
                optimizer= optimizer,
                scheduler=scheduler
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

        if hasattr(self.trainer, "reset_metrics"):
            self.trainer.reset_metrics()
            return

        if self.metrics:
            self.metrics.reset()

        if self.collector:
            self.collector.begin(
                self.params,
                step
            )

    #---------------------------------------------
    def on_validation_payload(self, payload):

        if hasattr(payload, "update_metrics"):
            payload.update_metrics(payload)
            return

        if self.metrics:
            self.metrics.update(payload)

        if self.collector:
            self.collector.update(payload)

    #---------------------------------------------
    def on_validation_end(self):

        if hasattr(
                self.trainer,
                "report_metrics",
        ):
            return self.trainer.report_metrics()

        summary = None
        if self.metrics:
            summary = self.metrics.summarize()

            if hasattr(self.metrics, "report"):
                summary = self.metrics.report(summary)

        if self.collector:
            self.collector.finalize()

        return summary