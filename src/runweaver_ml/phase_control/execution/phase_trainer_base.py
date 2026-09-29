from abc import ABC, abstractmethod

from .training_model_wrapper_base import system_state

class TrainerBase(ABC):
    def __init__(
            self,
            model,
            train_mode,
            eval_mode
    ):
        self.model = model
        self.TRAIN_MODE = train_mode
        self.EVAL_MODE = eval_mode

    @abstractmethod
    def compute(self, batch, params):
        raise NotImplementedError

    def train_context(self):
        return system_state(self.model, self.TRAIN_MODE)

    def eval_context(self):
        return system_state(self.model, self.EVAL_MODE)
