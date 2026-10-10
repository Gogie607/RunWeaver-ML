from abc import ABC, abstractmethod

from .transport_manifest import TransportManifest



class TransportBase(ABC):
    """Public extension contract for reiterable, synchronized sample access.

    Set ``manifest`` before supplying the transport to MultimodalDataset.
    A modality serializer of None means native Python values; other identifiers
    select FILETYPE_LOADERS. Samples contain normalized modality names,
    required ``__key__`` (stable identity), and optional ``__shard__`` (partition
    identity). Ordinary delivery does not require partitions. Requested partition
    operations and shard-aware providers may require __shard__ explicitly.
    Transports own synchronization and storage lifecycle.

    Each iter_samples call must start a fresh iteration. Honor requested
    deliverables, anchor selection and partition shuffling, or reject unsupported
    options explicitly. load_policy lists modalities eligible for deferred loading;
    eager delivery is allowed. DeferredPayload is resolved before middleware.
    Optional num_samples describes the default sample universe; otherwise MMDS
    counts raw samples. An explicit anchor causes a fresh raw count unless MMDS
    receives an explicit num_samples override.
    """

    manifest: TransportManifest
    num_samples: int | None = None

    TRANSPORT_NAME = "unknown"

    SUPPORTED_EXTENSIONS = ()


    @abstractmethod
    def iter_samples(
        self,
        *,
        deliverables,
        load_policy,
        shardshuffle=False,
        anchor_modality=None,
    ):
        pass


class DeferredPayload:

    def __init__(self, loader):
        self._loader = loader
        self._data = None
        self._loaded = False

    def load(self):

        if not self._loaded:
            if self._loader is None:
                raise RuntimeError("Deferred payload was closed or its load failed")
            try:
                self._data = self._loader()
                self._loaded = True
            finally:
                self.close()
        return self._data

    def close(self):
        """Abandon an unloaded payload, or release its completed/failed loader.

        Cached values remain usable. Failed loads are terminal; create a fresh
        payload to retry rather than retaining failed storage state.
        """
        loader = self._loader
        self._loader = None
        close = getattr(loader, "close", None)
        if close is not None:
            close()

    @property
    def loaded(self):
        return self._loaded
