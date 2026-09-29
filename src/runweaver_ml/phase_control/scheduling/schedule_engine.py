from __future__ import annotations

from typing import Any, Dict, Optional

from .schedules.schedule_classes import ScheduleBlock
from ..target_registry import TargetRegistry


class ScheduleRegistrationError(ValueError):
    pass



class ScheduleEngine:
    def __init__(self, registry:TargetRegistry, last_step: int = 0):
        self._registry = registry
        self._blocks: Dict[str, ScheduleBlock] = {}
        self._last_step = int(last_step)  # allow for checkpoint resumes

    def register(self, target:str, block: ScheduleBlock) -> None:
        if target in self._blocks:
            raise ScheduleRegistrationError(f"Target '{target}' already has a schedule")

        if not self._registry.has(target):
            raise ScheduleRegistrationError(f"Target '{target}' not registered")

        self._blocks[target] = block

        # initialize value
        value = block.value_at(self._last_step)
        self._registry.set(target, value)

    def update(self, step: int) -> None:
        self._last_step = int(step)
        for target, block in self._blocks.items():
            value = block.value_at(self._last_step)

            #type_name = block.__class__.__name__

            # optional debug/log
            # print(f"[Sched] {target}: {value:.4f} ({type_name})")
            self._registry.set(target, value)


    def is_registered(self, target: str) -> bool:
        return target in self._blocks

    def schedules(self) -> Dict[str, ScheduleBlock]:
        return dict(self._blocks)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "last_step": self._last_step,

        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        self._last_step = state.get("last_step", self._last_step)
