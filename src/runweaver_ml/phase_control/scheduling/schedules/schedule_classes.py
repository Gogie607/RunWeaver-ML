
from abc import ABC, abstractmethod
from typing import Any


import math

class ScheduleBlock(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def value_at(self, step: int) -> Any:
        raise NotImplementedError

    def __repr__(self):
        t = self.cfg.get("type", self.__class__.__name__)
        params = ", ".join(
            f"{k}={v}"
            for k, v in self.cfg.items()
            if k != "type"
        )
        return f"{t}({params})"

###########################################
#   LinearSchedule
#   Desc.  linear increment/decrement across active window
#
###########################################
"""
    Required parameters in config file schedule definition
        target_name:
          type: linear
          start: value assigned on entering active window
          end: final value at end of activation window
          start_step:  start of activation window
          end_step:  end of activation window
"""
class LinearSchedule(ScheduleBlock):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def value_at(self, step):
        if step <= self.cfg["start_step"]:
            return self.cfg["start"]
        if self.cfg["end_step"] == self.cfg["start_step"]:
            return self.cfg["end"]

        if  step >= self.cfg["end_step"]:
            return self.cfg["end"]

        t = (step - self.cfg["start_step"]) / (self.cfg["end_step"] - self.cfg["start_step"])
        return self.cfg["start"] + t * (self.cfg["end"] - self.cfg["start"])


###########################################
#  CosineSchedule
#   Desc.  decaying increment/decrement across active window
#
###########################################
"""
    Required parameters in config file schedule definition
        target_name:
          type: linear
          start: value assigned on entering active window
          end: final value at end of activation window
          start_step:  start of activation window
          end_step:  end of activation window
"""
class CosineSchedule(ScheduleBlock):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def value_at(self, step):
        if step <= self.cfg["start_step"]:
            return self.cfg["start"]

        if self.cfg["end_step"] == self.cfg["start_step"]:
            return self.cfg["end"]

        if self.cfg["end_step"] is None or step >= self.cfg["end_step"]:
            return self.cfg["end"]

        t = (step - self.cfg["start_step"]) / (self.cfg["end_step"] - self.cfg["start_step"])
        cos_t = 0.5 * (1 - math.cos(math.pi * t))
        return self.cfg["start"] + cos_t * (self.cfg["end"] - self.cfg["start"])


###########################################
#  StepSchedule
#   Desc.  change  target value at given step
#
###########################################
"""
    Required parameters in config file schedule definition
        target_name:
          type: linear
          start: value assigned on entering active window
          end: final value at end of activation window
          start_step:  start of activation window
          end_step:  end of activation window
"""
class StepSchedule(ScheduleBlock):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def value_at(self, step):
        return self.cfg["value"] if step >= self.cfg["at_step"] else self.cfg["start"]


###########################################
#
###########################################
""""""
class GateSchedule(ScheduleBlock):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def value_at(self, step):
        if self.cfg["start_step"] <= step <= self.cfg["end_step"]:
            return self.cfg["value"]
        return self.cfg["start"]

###########################################
#
###########################################
"""
    Required parameters in config file schedule definition
        target_name:
          type: square
          high:  'value'
          low:  'value'
          period:  'value'  number of steps defining a cycle
          duty:  'value'  % of cycle in high state
"""
class SquareSchedule(ScheduleBlock):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def value_at(self, step):
        phase = step % self.cfg["period"]
        return self.cfg["high"] if phase < self.cfg["period"] * self.cfg["duty"] else self.cfg["low"]



