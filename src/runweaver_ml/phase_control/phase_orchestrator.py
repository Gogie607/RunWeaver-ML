
import torch

from  .train_loop_config import TrainLoopConfig, AmpSettings

from .execution import TrainingModule

class PhaseOrchestrator:

    def __init__(self, model):
        self.model = model

    def run(self, phases):
        for i, ph in enumerate(phases):
            print(f"\n=== Phase {i}: {ph['name']} ===\n")

            #self._apply_model_freeze(ph.get("model", {}))
            #self._configure_optim(ph.get("optimizer", {}))
            #loader = self._configure_loader(ph.get("data", {}))

            # build a *phase-local* training spec (full, not delta)
            #train_spec = {
            #    "loss": ph["loss"],
            #    "steps": ph["steps"],
            #    "name": ph["name"],
            #}
            # TODO define how to use if possible
            # run your existing loop unchanged
            #train_loop(
            #    model=self.model,
            #    loader=loader,
            #    optimizer=self.optimizer,
            #    scheduler=self.scheduler,
            #    spec=train_spec,
            #    start_step=self.global_step,
            #    device=self.device,
            #)

            #self.global_step += ph["steps"]

    @staticmethod
    def train_epoch(
            trainer :TrainingModule,
            loop_config:TrainLoopConfig,
            optimizer,
            train_loader,
            val_loader=None,
            *,
            logger=None,
            start_step=0,
            scheduler = None,
    ):

        step = start_step

        trainer.on_train_begin()


        use_scaler = loop_config.amp.use_scaler

        scaler = torch.amp.GradScaler('cuda',enabled=use_scaler)

        with (trainer.train_context() as tc):

            for batch in train_loader:
                step += 1

                #params.step(step)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                        'cuda',
                        enabled=loop_config.amp.enabled,
                        dtype=loop_config.amp.dtype,
                ):
                    trainer.on_step_begin(step)
                    loss, payload = trainer.compute(batch)

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                trainer.on_train_payload(payload)

                if ( loop_config.log_every is not None
                     and step %  loop_config.log_every == 0
                ):
                    try:
                        if logger:
                            logger.log(
                                tag="train_step",
                                step=step,
                                data= [trainer.on_log(payload),
                                       f" LR: {optimizer.param_groups[0]['lr']:.2e}"]
                            )
                    except TypeError as e:
                        print(f"Logger returned a Type error: {e}")


                # ---- validation hook ----
                if (
                        loop_config.validate_every is not None
                        and val_loader is not None
                        and step % loop_config.validate_every == 0
                ):
                    if logger:
                        logger.flush()

                    trainer.on_validation_begin( step)

                    metric_obj = PhaseOrchestrator.validate_epoch(
                        trainer,
                        loop_config,
                        val_loader
                    )

                    if logger:
                        logger.log(
                            tag="validation",
                            step=step,
                            data= metric_obj
                        )
                        logger.flush()

                if (  loop_config.save_every is not None
                    and (step % loop_config.save_every) == 0
                ):
                    trainer.on_checkpoint( optimizer, scheduler)

                if ( loop_config.max_steps is not None
                    and step >= loop_config.max_steps
                ):
                    break

        return step


    @torch.no_grad()
    @staticmethod
    def validate_epoch(
            trainer : TrainingModule,
            loop_config : TrainLoopConfig,
            loader
    ) ->dict:

        with trainer.eval_context()as tc:

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

