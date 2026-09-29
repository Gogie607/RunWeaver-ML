from abc import ABC, abstractmethod

"""
Core datasource execution abstractions.

These classes define the operational execution graph used by
DatasetView orchestration.

The system separates execution nodes into two major categories:

------------------------------------------------------------
1. IteratableRootBase
------------------------------------------------------------

Root stream originators.

These classes CREATE the iterable stream and therefore define:
    - iteration lifecycle
    - synchronization
    - sample boundaries
    - transport topology

Examples:
    - LayoutSource
    - ArrowSource
    - SQLSource
    - RandomSource

Root sources are responsible for bootstrapping execution.

Without at least one root source:
    no iterable stream exists.

Execution entrypoint:
    __iter__()

------------------------------------------------------------
2. MiddlewareSourceBase
------------------------------------------------------------

Stream transformers / augmenters.

These classes DO NOT create streams.
Instead, they consume an existing stream and emit a modified one.

Typical responsibilities:
    - artifact injection
    - filtering
    - decoding
    - augmentation
    - normalization
    - batching helpers

Examples:
    - ProviderBase
    - ProsodyProvider
    - FilteringProvider

Execution entrypoint:
    iterate(stream)

------------------------------------------------------------
Negotiation System
------------------------------------------------------------

All datasource nodes inherit NegotiableBase.

Negotiation allows DatasetView to determine:

    capabilities:
        What COULD this node provide?

    request():
        What WILL this node provide for the
        current execution request?

    deliverables:
        Which requested items are currently active?

    requires:
        Intrinsic immutable dependencies.

Example:
    ProsodyGenerator.requires -> ["wav"]

------------------------------------------------------------
Execution Flow
------------------------------------------------------------

Typical execution chain:

    RootSource
        -> MiddlewareSource
        -> MiddlewareSource
        -> yield sample

Example:

    LayoutSource
        -> NormProvider
        -> AugmentationProvider
        -> training loader

DatasetView acts as the execution orchestrator responsible for:
    - negotiation
    - pipeline activation
    - execution chaining

------------------------------------------------------------
Design Philosophy
------------------------------------------------------------

Root sources:
    originate iterable execution.

Middleware sources:
    transform iterable execution.

DatasetView:
    composes the operational graph.

These abstractions intentionally avoid coupling to:
    - WebDataset
    - Arrow
    - Torch
    - storage format

allowing alternative transport and execution systems to be
implemented without changing downstream pipeline logic.
"""

# Base class for all data suppliers  describing their state
class NegotiableBase:

    def __init__(self):

        self._capabilities = []
        self._deliverables = []

    # =====================================================
    # CAPABILITIES
    # =====================================================

    @property
    def capabilities(self) ->list[str]:
        """
        Answers the question 'What COULD I produce if requested'

        Static capability declaration.
        Independent of runtime requests.

        Subclasses must populate this member properly
        or override to provide unique behavior
        """
        return self._capabilities

    # =====================================================
    # DELIVERABLES
    # =====================================================

    @property
    def deliverables(self):
        """
        Answers the question 'What requests am I currently fulfilling

        Defines dynamically a set of capabilities
        specifically requested and being emitted.

        Subclasses must populate this member properly
        or override to provide unique behavior
        """

        return self._deliverables

    # =====================================================
    # DEPENDENCIES
    # =====================================================

    @property
    def requires(self)->list[str]:
        """
        Answers the question "What do I need?

        Intrinsic immutable dependencies.

        Example:
            ProsodyGenerator -> ["wav"]

        Composite sources may return [].
        """
        return []


    # =====================================================
    # NEGOTIATION
    # =====================================================

    @abstractmethod
    def request(
            self,
            request,

    )-> list[str]:
        """
        Total items request.
        Answers the question "What WILL I offer?

        Actively fulfills the as many items in request
        that are listen in capabilities

        Populates:
            self._deliverables (fulfilled requests)

        Returns:
            list of fulfilled requests
        Note:
            implementation currently has the option of:
             1. invalidating current active requests
             2. accumulating multiple request

        """
        raise NotImplementedError()


class MiddlewareSourceBase(NegotiableBase):

    # =====================================================
    # EXECUTION
    # =====================================================

    @abstractmethod
    def iterate(
        self,
        stream,
        ctx
    ):
        raise NotImplementedError()



class IteratableRootBase(NegotiableBase):

    # =====================================================
    # EXECUTION
    # =====================================================

    @abstractmethod
    def __iter__(
        self,
    ):
        raise NotImplementedError()