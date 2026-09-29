"""Model wrapper interface required by phase-control execution."""

from contextlib import contextmanager
from typing import Any


class TrainingModelWrapperBase:
    """Interface for external AI model encapsulations used by trainers.

    Implementations are expected to expose structured mode and trainability
    state so phase-control code can temporarily switch execution state without
    knowing model internals.
    """

    def get_mode(self) -> Any:
        ...

    def set_mode(self, state: Any):
        ...

    def get_trainable(self) -> Any:
        ...

    def set_trainable(self, state: Any):
        ...

    def to(self, device):
        return self


@contextmanager
def system_state(system: TrainingModelWrapperBase, new_mode):
    """Temporarily switch a wrapped model to ``new_mode``."""
    mode = system.get_mode()
    try:
        system.set_mode(new_mode)
        yield
    finally:
        system.set_mode(mode)


@contextmanager
def system_trainable(system: TrainingModelWrapperBase, trainables):
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
