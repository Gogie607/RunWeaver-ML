from abc import ABC, abstractmethod



class TransportBase(ABC):

    TRANSPORT_NAME = "unknown"

    SUPPORTED_EXTENSIONS = ()


    @abstractmethod
    def iter_samples(
        self,
        *,
        deliverables,
        load_policy,
        shardshuffle=False,
    ):
        pass


class DeferredPayload:

    def __init__(self, loader):
        self._loader = loader
        self._data = None

    def load(self):

        if self._data is None:
            self._data = self._loader()
            self._loader = None   # release closure / tar references
        return self._data

    @property
    def loaded(self):
        return self._data is not None
