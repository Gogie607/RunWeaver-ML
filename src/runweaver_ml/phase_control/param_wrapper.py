#
#   A thin Facade encapsulating
#   ParamProvider and schedule-engine
#
#
from .scheduling.schedule_engine import ScheduleEngine
from .param_provider import ParamProvider
from contextlib import contextmanager



class ParameterWrapper:
    def __init__(self,
                 engine:ScheduleEngine,
                 params:ParamProvider):

        self._engine = engine
        self._params = params

    # step the schedule-engine to update dynamic parameters
    def step(self, step: int):
        self._engine.update(step)

    def get(self, key):
        return self._params.get(key)

    def has(self, key):
        return self._params.has(key)

    def check_required(self, keys:list[str]):
        return [k for k in keys if not self.has(k)]


    def dump(self):
        return self._params.registry.as_dict()

    @property
    def params(self):
        return self._params
    @property
    def engine(self):
        return self._engine

    # run contains static parameters used during training
    # current members expected to be here
    # steps = training steps, trainable = unfrozen model components
