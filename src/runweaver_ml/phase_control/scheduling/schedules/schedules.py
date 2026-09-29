
from .schedule_classes import *

######################################################


def build_schedule_block(cfg: dict):
    t = cfg["type"]


    if t == "linear":
        return LinearSchedule(cfg)

    elif t == "cosine":
        return CosineSchedule(cfg)


    elif t == "step":
        return StepSchedule(cfg)

    elif t == "gate":
        return GateSchedule(cfg)

    elif t == "square":
        return SquareSchedule(cfg)
    else:
        raise RuntimeError(f"Unknown schedule type: {t}")

#############################################################