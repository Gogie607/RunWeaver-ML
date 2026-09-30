"""Training phase orchestration and scheduled parameter control."""

from .execution import (
    CollectorBase,
    MetricHandlerBase,
    TrainerBase,
    TrainingModelProtocol,
    TrainingModelWrapperBase,
    TrainingModule,
    system_state
)
from .param_provider import ParamProvider
from .param_wrapper import ParameterWrapper
from .phase_orchestrator import PhaseOrchestrator
from .objectives import (
    #ComposedResult,
    ForwardStrategy,
    ObjectiveComposer,
    TrainingContext,
    TrainingObjective,
)
from .scheduling import ScheduleEngine
from .scheduling.schedules import build_schedule_block
from .target_registry import TargetRegistry
from .train_phase_manager import PhaseContext, PhaseManager
from .train_loop_config import  TrainLoopConfig, AmpSettings
__all__ = [
    "CollectorBase",
    "MetricHandlerBase",
    "ParamProvider",
    "ParameterWrapper",
    "PhaseContext",
    "PhaseManager",
    "PhaseOrchestrator",
    #"ComposedResult",
    "ForwardStrategy",
    "ObjectiveComposer",
    "ScheduleEngine",
    "TargetRegistry",
    "TrainingContext",
    "TrainingObjective",
    "TrainerBase",
    "TrainingModelWrapperBase",
    "TrainingModelProtocol",
    "TrainingModule",
    "system_state",
    "build_schedule_block",
    "TrainLoopConfig",
    "AmpSettings",
]
