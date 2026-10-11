import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from runweaver_ml.phase_control import PhaseOrchestrator, TrainLoopConfig


class ScalarTrainer:
    """Tiny differentiable trainer recording the existing batch-based hooks."""

    def __init__(self, device='cpu'):
        self.model = torch.nn.Linear(1, 1, bias=False, device=device)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        self.steps = []
        self.losses = []
        self.checkpoints = []
        self.ended = False

    def train_context(self):
        return nullcontext()

    def on_train_begin(self):
        pass

    def on_train_end(self):
        self.ended = True

    def on_step_begin(self, step):
        self.steps.append(step)

    def compute(self, batch):
        loss = self.model(batch).square().mean()
        return loss, float(loss.detach())

    def on_train_payload(self, payload):
        self.losses.append(payload)

    def on_checkpoint(self, optimizer, scheduler):
        self.checkpoints.append((self.steps[-1], float(self.model.weight.detach())))


class GradientAccumulationTest(unittest.TestCase):
    def config(self, *, count=None, steps=5, save_every=None, amp=None):
        runtime = {'schedule': {'save_every': save_every}}
        if count is not None:
            runtime['gradient_accumulation'] = {'enabled': True, 'count': count}
        if amp is not None:
            runtime['amp'] = amp
        return TrainLoopConfig.from_config(runtime, max_steps=steps)

    def run_loop(self, trainer, config, batches):
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        with patch.object(optimizer, 'step', wraps=optimizer.step) as updates:
            final_step = PhaseOrchestrator.train_epoch(
                trainer, config, optimizer, batches, scheduler=scheduler,
            )
        return final_step, updates.call_count, scheduler

    def reference(self, values, count, device='cpu', amp=None):
        model = torch.nn.Linear(1, 1, bias=False, device=device)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        use_scaler = amp is not None and amp['dtype'] == 'float16'
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler, init_scale=128)
        dtype = torch.float16 if use_scaler else torch.bfloat16
        for offset in range(0, len(values), count):
            optimizer.zero_grad(set_to_none=True)
            batch = torch.tensor(values[offset:offset + count], device=device).reshape(-1, 1)
            with torch.amp.autocast('cuda', enabled=amp is not None, dtype=dtype):
                loss = model(batch).square().mean()
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
        return model.weight.detach()

    def test_absent_or_disabled_config_preserves_per_batch_updates(self):
        for settings in [{}, {'enabled': False}, {'count': 4}]:
            with self.subTest(settings=settings):
                config = TrainLoopConfig.from_config(
                    {'gradient_accumulation': settings}, max_steps=3,
                )
                self.assertEqual(config.accumulation_steps, 1)
        self.assertEqual(TrainLoopConfig.from_config({}, max_steps=3).accumulation_steps, 1)
        trainer = ScalarTrainer()
        final_step, updates, _ = self.run_loop(
            trainer, self.config(steps=3), [torch.ones(1, 1)] * 3,
        )
        self.assertEqual((final_step, updates), (3, 3))

    def test_enabled_config_requires_count(self):
        with self.assertRaisesRegex(ValueError, 'count is required'):
            TrainLoopConfig.from_config({'gradient_accumulation': {'enabled': True}}, max_steps=1)

    def test_invalid_counts_and_enable_flags_are_rejected(self):
        for value in [None, True, False, 0, -1, 1.5, '2']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                TrainLoopConfig(
                    max_steps=1, validate_every=None, validate_max_steps=None,
                    save_every=None, log_every=None, amp=self.config().amp,
                    accumulation_steps=value,
                )
            with self.subTest(config_value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                TrainLoopConfig.from_config(
                    {'gradient_accumulation': {'enabled': True, 'count': value}}, max_steps=1,
                )
        for value in ['true', 1, None]:
            with self.subTest(enabled=value), self.assertRaisesRegex(ValueError, 'boolean'):
                TrainLoopConfig.from_config({'gradient_accumulation': {'enabled': value}}, max_steps=1)
        with self.assertRaisesRegex(ValueError, 'dictionary'):
            TrainLoopConfig.from_config({'gradient_accumulation': True}, max_steps=1)

    def test_accumulated_update_matches_combined_batch_and_short_tail(self):
        values = [1., 2., 3., 4., 5.]
        trainer = ScalarTrainer()
        final_step, updates, scheduler = self.run_loop(
            trainer, self.config(count=2), [torch.tensor([[value]]) for value in values],
        )
        torch.testing.assert_close(trainer.model.weight, self.reference(values, 2))
        self.assertEqual((final_step, updates, scheduler.last_epoch), (5, 3, 3))
        self.assertEqual(trainer.steps, [1, 2, 3, 4, 5])
        self.assertEqual(trainer.losses[:2], [1., 4.])  # Payload losses are unscaled.
        self.assertTrue(trainer.ended)

    def test_step_limit_flushes_partial_group(self):
        trainer = ScalarTrainer()
        values = [1., 2., 3., 4., 5., 6.]
        final_step, updates, _ = self.run_loop(
            trainer, self.config(count=3, steps=2), [torch.tensor([[value]]) for value in values],
        )
        torch.testing.assert_close(trainer.model.weight, self.reference(values[:2], 3))
        self.assertEqual((final_step, updates), (2, 1))

    def test_empty_loader_does_not_update(self):
        trainer = ScalarTrainer()
        final_step, updates, scheduler = self.run_loop(trainer, self.config(count=2), [])
        self.assertEqual((final_step, updates, scheduler.last_epoch), (0, 0, 0))
        self.assertTrue(trainer.ended)

    def test_checkpoints_wait_for_updates_including_tail(self):
        trainer = ScalarTrainer()
        self.run_loop(trainer, self.config(count=3, save_every=1), [torch.ones(1, 1)] * 5)
        self.assertEqual([step for step, weight in trainer.checkpoints], [3, 5])
        self.assertNotEqual(trainer.checkpoints[0][1], trainer.checkpoints[1][1])

    def test_exception_discards_pending_gradients_and_runs_end_hook(self):
        trainer = ScalarTrainer()
        original_compute = trainer.compute

        def fail_on_second(batch):
            if len(trainer.steps) == 2:
                raise RuntimeError('failed compute')
            return original_compute(batch)

        trainer.compute = fail_on_second
        with self.assertRaisesRegex(RuntimeError, 'failed compute'):
            self.run_loop(trainer, self.config(count=3), [torch.ones(1, 1)] * 3)
        self.assertTrue(trainer.ended)
        self.assertIsNone(trainer.model.weight.grad)
        self.assertEqual(float(trainer.model.weight.detach()), 1.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for real AMP integration')
    def test_real_cuda_amp_accumulation_matches_combined_batches(self):
        for dtype in ['float16', 'bfloat16']:
            with self.subTest(dtype=dtype):
                amp = {'enabled': True, 'dtype': dtype}
                trainer = ScalarTrainer('cuda')
                values = [1., 2., 3., 4., 5.]
                # Low scale avoids incidental overflow in this equivalence test.
                real_scaler = torch.amp.GradScaler
                with patch(
                    'runweaver_ml.phase_control.phase_orchestrator.torch.amp.GradScaler',
                    side_effect=lambda *args, **kwargs: real_scaler(*args, **kwargs, init_scale=128),
                ):
                    final_step, updates, scheduler = self.run_loop(
                        trainer, self.config(count=2, amp=amp),
                        [torch.tensor([[value]], device='cuda') for value in values],
                    )
                torch.testing.assert_close(
                    trainer.model.weight, self.reference(values, 2, 'cuda', amp), atol=0.005, rtol=0.01,
                )
                self.assertEqual((final_step, updates, scheduler.last_epoch), (5, 3, 3))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for GradScaler overflow')
    def test_amp_overflow_skips_optimizer_and_scheduler_then_recovers(self):
        trainer = ScalarTrainer('cuda')
        batches = [torch.tensor([[float('inf')]], device='cuda')]
        batches += [torch.ones(1, 1, device='cuda')] * 3
        final_step, updates, scheduler = self.run_loop(
            trainer, self.config(count=2, steps=4, amp={'enabled': True, 'dtype': 'float16'}), batches,
        )
        self.assertEqual((final_step, updates, scheduler.last_epoch), (4, 1, 1))
        self.assertTrue(torch.isfinite(trainer.model.weight).all())
        self.assertLess(float(trainer.model.weight.detach()), 1.)


if __name__ == '__main__':
    unittest.main()
