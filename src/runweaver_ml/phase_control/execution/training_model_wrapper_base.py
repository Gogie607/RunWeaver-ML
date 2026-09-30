"""Model wrapper interface required by phase-control execution."""

from contextlib import contextmanager
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from torch.nn import Parameter


@runtime_checkable
class TrainingModelProtocol(Protocol):
    """Capabilities required by phase-control execution.

    Concrete model construction, serialization, and persistence remain owned by
    the application. Implementations satisfy this protocol structurally and do
    not need to inherit from a RunWeaver class.
    """

    def get_mode(self) -> Any:
        ...

    def set_mode(self, state: Any):
        ...

    def get_trainable(self) -> Any:
        ...

    def set_trainable(self, state: Any):
        ...

    def trainable_parameters(self) -> Iterable[Parameter]:
        ...

    def to(self, device):
        ...


# Compatibility alias for applications that imported the former nominal name.
TrainingModelWrapperBase = TrainingModelProtocol


@contextmanager
def system_state(system: TrainingModelProtocol, new_mode):
    """Temporarily switch a wrapped model to ``new_mode``."""
    mode = system.get_mode()
    try:
        system.set_mode(new_mode)
        yield
    finally:
        system.set_mode(mode)


@contextmanager
def system_trainable(system: TrainingModelProtocol, trainables):
    """Temporarily restrict trainable leaves on a wrapped model."""
    old = system.get_trainable()

    state = clear_trainable(system.get_trainable())
    state = apply_trainable(state, trainables)
    system.set_trainable(state)

    try:
        yield
    finally:
        system.set_trainable(old)


def apply_trainable(state, trainables):
    for item in trainables:
        parts = item.split(".")

        node = state
        for part in parts[:-1]:
            node = node[part]

        node[parts[-1]] = True

    return state


def clear_trainable(state):
    """Recursively set every trainable flag to False while preserving shape."""
    if isinstance(state, bool):
        return False

    if isinstance(state, dict):
        return {
            key: clear_trainable(value)
            for key, value in state.items()
        }

    return state
