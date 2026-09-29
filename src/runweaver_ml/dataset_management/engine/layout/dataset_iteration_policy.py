
from dataclasses import dataclass

from enum import Enum, auto


class MixMode(Enum):
    MIXED = auto()
    PER_DOMAIN = auto()

class ExhaustPolicy(Enum):
    LOOP = auto()                    # stationary infinite reloads on exhaustion
    FINITE_EXHAUSTIVE = auto()    # consume all samples then stop
    FINITE_STATIONARY = auto()    # stop on first exhaustion
    FINITE = FINITE_EXHAUSTIVE    # default finite mode


@dataclass
class DatasetIterationPolicy:

    mix_mode: MixMode = MixMode.PER_DOMAIN

    exhaust_policy: ExhaustPolicy = (
        ExhaustPolicy.FINITE
    )

    @classmethod
    def from_config(cls, cfg):
        cfg = cfg or {}
        return cls(
            mix_mode=MixMode[
                cfg.get(
                    "mix_mode",
                    "per_domain"
                ).upper()
            ],

            exhaust_policy=ExhaustPolicy[
                cfg.get(
                    "exhaust_policy",
                    "finite"
                ).upper()
            ],
        )



