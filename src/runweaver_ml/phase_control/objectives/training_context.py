from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_MISSING = object()


@dataclass
class TrainingContext:
    """Shared per-batch workspace for objective-based training."""

    batch: Any
    values: dict[str, Any] = field(default_factory=dict)

    def put(self, name: str, value: Any) -> None:
        self.values[name] = value

    def get(self, name: str, default: Any = _MISSING) -> Any:
        if name in self.values:
            return self.values[name]

        if default is not _MISSING:
            return default

        raise KeyError(name)

    def has(self, name: str) -> bool:
        return name in self.values

    def require(self, *names: str) -> None:
        missing = [
            name
            for name in names
            if name not in self.values
        ]

        if missing:
            raise RuntimeError(
                f"Training context missing required values: {missing}"
            )
