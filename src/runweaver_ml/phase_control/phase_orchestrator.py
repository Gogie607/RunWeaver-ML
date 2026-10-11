
import torch

from .train_loop_config import TrainLoopConfig

from .execution import TrainingModule

class PhaseOrchestrator:

    @staticmethod
    def train_epoch(
            trainer :TrainingModule,
            loop_config:TrainLoopConfig,
            optimizer,
            train_loader,
            val_loader=None,
            *,
            logger=None,
            scheduler = None,
    ):

        step = 0
        pending_batches = 0
        checkpoint_pending = False
        accumulation_steps = loop_config.accumulation_steps

        def apply_update(batch_count):
            # Every backward uses the configured divisor. A final short group
            # needs correction to average only the batches actually consumed.
            if batch_count != accumulation_steps:
                correction = accumulation_steps / batch_count
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)

            updated = True
            if use_scaler:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler skips optimizer.step on nonfinite gradients.
                updated = scaler.get_scale() >= previous_scale
            else:
                optimizer.step()
            if scheduler is not None and updated:
                scheduler.step()

        trainer.on_train_begin()
        try:
            use_scaler = loop_config.amp.use_scaler
            scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

            with trainer.train_context():
                for batch in train_loader:
                    if (
                        loop_config.max_steps is not None
                        and step >= loop_config.max_steps
                    ):
                        break

                    step += 1
                    if pending_batches == 0:
                        optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast(
                            'cuda',
                            enabled=loop_config.amp.enabled,
                            dtype=loop_config.amp.dtype,
                    ):
                        trainer.on_step_begin(step)
                        loss, payload = trainer.compute(batch)

                    pending_batches += 1
                    if use_scaler:
                        scaler.scale(loss / accumulation_steps).backward()
                    else:
                        (loss / accumulation_steps).backward()

                    if pending_batches == accumulation_steps:
                        apply_update(pending_batches)
                        pending_batches = 0

                    trainer.on_train_payload(payload)

                    if (
                        loop_config.log_every is not None
                        and step % loop_config.log_every == 0
                        and logger
                    ):
                        logger.log(
                            tag="train_step",
                            step=step,
                            data=[
                                trainer.on_log(payload),
                                f" LR: {optimizer.param_groups[0]['lr']:.2e}",
                            ],
                        )

                    if (
                        loop_config.validate_every is not None
                        and val_loader is not None
                        and step % loop_config.validate_every == 0
                    ):
                        if logger:
                            logger.flush()

                        metric_obj = PhaseOrchestrator.validate_epoch(
                            trainer,
                            loop_config,
                            val_loader,
                            step=step,
                        )

                        if logger:
                            logger.log(
                                tag="validation",
                                step=step,
                                data=metric_obj,
                            )
                            logger.flush()

                    if (
                        loop_config.save_every is not None
                        and step % loop_config.save_every == 0
                    ):
                        checkpoint_pending = True
                    if checkpoint_pending and pending_batches == 0:
                        trainer.on_checkpoint(optimizer, scheduler)
                        checkpoint_pending = False

                if pending_batches:
                    apply_update(pending_batches)
                    pending_batches = 0
                    if checkpoint_pending:
                        trainer.on_checkpoint(optimizer, scheduler)
        finally:
            try:
                # Do not carry an unfinished group into another phase after an
                # exception. Successful final groups were applied above.
                if pending_batches:
                    optimizer.zero_grad(set_to_none=True)
            finally:
                trainer.on_train_end()

        return step


    @torch.no_grad()
    @staticmethod
    def validate_epoch(
            trainer : TrainingModule,
            loop_config : TrainLoopConfig,
            loader,
            *,
            step: int = 0,
    ) ->dict:

        trainer.on_validation_begin(step)

        with trainer.eval_context():

            for local_step, batch in enumerate(loader):
                if loop_config.validate_max_steps is not None and local_step >= loop_config.validate_max_steps:
                    break

                with torch.amp.autocast(
                        'cuda',
                        enabled=loop_config.amp.enabled,
                        dtype=loop_config.amp.dtype,
                ):
                    _, payload = trainer.compute(batch)

                trainer.on_validation_payload(payload)


        summary = trainer.on_validation_end()
        return summary
