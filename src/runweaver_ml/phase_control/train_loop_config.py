from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AmpSettings:
    enabled: bool
    dtype: torch.dtype
    use_scaler: bool

    @classmethod
    def from_config(cls, config: dict | None) -> "AmpSettings":
        config = config or {}

        dtype_name = config.get("dtype", "float16")

        if dtype_name == "float16":
            dtype = torch.float16
        elif dtype_name == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported AMP dtype: {dtype_name}")

        enabled = bool(config.get("enabled", False)) and torch.cuda.is_available()

        return cls(
            enabled=enabled,
            dtype=dtype,
            use_scaler=(
                enabled
                and dtype is torch.float16
                and bool(config.get("grad_scaler", True))
            ),
        )


@dataclass(frozen=True)
class TrainLoopConfig:
    max_steps: int
    validate_every: int | None
    validate_max_steps: int | None
    save_every: int | None
    log_every: int | None
    amp: AmpSettings

    @classmethod
    def from_config(
        cls,
        runtime_config: dict,
        *,
        max_steps: int,
    ) -> "TrainLoopConfig":
        schedule = runtime_config.get("schedule", {})

        values = {
            "max_steps": max_steps,
            "validate_every": schedule.get("validate_every"),
            "validate_max_steps": schedule.get("validate_max_steps"),
            "save_every": schedule.get("save_every"),
            "log_every": schedule.get("log_every"),
        }
        for name, value in values.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")
            if name != "max_steps" and value == 0:
                raise ValueError(f"{name} must be positive or None")

        return cls(
            max_steps=values["max_steps"],
            validate_every=values["validate_every"],
            validate_max_steps=values["validate_max_steps"],
            save_every=values["save_every"],
            log_every=values["log_every"],
            amp=AmpSettings.from_config(runtime_config.get("amp")),
        )
