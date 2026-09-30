import unittest
from contextlib import contextmanager

import torch

from runweaver_ml.phase_control import (
    AmpSettings,
    ParamProvider,
    PhaseOrchestrator,
    TargetRegistry,
    TrainLoopConfig,
    TrainingModule,
)


class RecordingParams:
    def __init__(self):
        self.steps = []

    def step(self, step):
        self.steps.append(step)

    def get(self, key):
        return None


class RecordingTrainer:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.events = []
        self.metric_updates = []

    @contextmanager
    def train_context(self):
        self.events.append("train_context_begin")
        try:
            yield
        finally:
            self.events.append("train_context_end")

    @contextmanager
    def eval_context(self):
        self.events.append("eval_context_begin")
        try:
            yield
        finally:
            self.events.append("eval_context_end")

    def compute(self, batch, params):
        prediction = self.model(batch)
        return prediction.square().mean(), {"batch": batch.detach().clone()}

    def reset_metrics(self):
        self.events.append("metrics_reset")
        self.metric_updates.clear()

    def update_metrics(self, payload):
        self.events.append("metrics_update")
        self.metric_updates.append(payload)

    def report_metrics(self):
        self.events.append("metrics_report")
        return {"batches": len(self.metric_updates)}


class RecordingModule(TrainingModule):
    def on_train_begin(self):
        self.trainer.events.append("train_begin")

    def on_train_end(self):
        self.trainer.events.append("train_end")


def loop_config(*, steps=2, validate_every=None, validate_max_steps=None, save_every=None):
    return TrainLoopConfig(
        max_steps=steps,
        validate_every=validate_every,
        validate_max_steps=validate_max_steps,
        save_every=save_every,
        log_every=None,
        amp=AmpSettings(
            enabled=False,
            dtype=torch.float16,
            use_scaler=False,
        ),
    )


class PhaseExecutionContractTest(unittest.TestCase):
    def test_steps_are_phase_local_and_lifecycle_finishes(self):
        trainer = RecordingTrainer()
        params = RecordingParams()
        module = RecordingModule(trainer, params)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)

        final_step = PhaseOrchestrator.train_epoch(
            module,
            loop_config(steps=2),
            optimizer,
            [torch.ones(1, 1) for _ in range(4)],
        )

        self.assertEqual(final_step, 2)
        self.assertEqual(params.steps, [1, 2])
        self.assertEqual(trainer.events[0], "train_begin")
        self.assertEqual(trainer.events[-1], "train_end")

    def test_train_end_runs_when_compute_raises(self):
        trainer = RecordingTrainer()
        params = RecordingParams()
        module = RecordingModule(trainer, params)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)

        def fail(batch, params):
            raise RuntimeError("failed compute")

        trainer.compute = fail
        with self.assertRaisesRegex(RuntimeError, "failed compute"):
            PhaseOrchestrator.train_epoch(
                module,
                loop_config(steps=1),
                optimizer,
                [torch.ones(1, 1)],
            )

        self.assertEqual(trainer.events[-1], "train_end")

    def test_validation_lifecycle_is_owned_by_validate_epoch(self):
        trainer = RecordingTrainer()
        module = RecordingModule(
            trainer,
            RecordingParams(),
            validation_owner="trainer",
        )

        summary = PhaseOrchestrator.validate_epoch(
            module,
            loop_config(steps=1, validate_max_steps=1),
            [torch.ones(1, 1), torch.ones(1, 1)],
            step=7,
        )

        self.assertEqual(summary, {"batches": 1})
        self.assertEqual(
            trainer.events,
            [
                "metrics_reset",
                "eval_context_begin",
                "metrics_update",
                "eval_context_end",
                "metrics_report",
            ],
        )

    def test_checkpointing_uses_only_injected_handler(self):
        calls = []
        trainer = RecordingTrainer()
        module = RecordingModule(
            trainer,
            RecordingParams(),
            checkpoint_handler=lambda module, optimizer, scheduler: calls.append(
                (module.current_step, scheduler)
            ),
        )
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)

        PhaseOrchestrator.train_epoch(
            module,
            loop_config(steps=1, save_every=1),
            optimizer,
            [torch.ones(1, 1)],
        )

        self.assertEqual(calls, [(1, None)])

    def test_zero_intervals_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "log_every must be positive"):
            TrainLoopConfig.from_config(
                {"schedule": {"log_every": 0}},
                max_steps=1,
            )

    def test_nested_parameter_lookup_preserves_layer_precedence(self):
        provider = ParamProvider(
            run={"section": {"value": "run"}},
            global_cfg={"section": {"value": "global"}},
            runtime={"section": {"value": "runtime"}},
            registry=TargetRegistry({}),
        )

        self.assertTrue(provider.has("value"))
        self.assertEqual(provider.get("value"), "run")


if __name__ == "__main__":
    unittest.main()
