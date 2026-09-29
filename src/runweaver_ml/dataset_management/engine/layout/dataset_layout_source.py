import random
from itertools import groupby

from dataclasses import dataclass

from ...source.datasource_base import IteratableRootBase
from .filetype_registry import FILETYPE_LOADERS


from ..transport.transport_base import (
    DeferredPayload
)

# required parameter encapsulation object for LayoutSource
@dataclass
class LayoutOpParams:

    shuffle: bool = False
    shardshuffle: bool = False
    shuffle_buffer: int = 1000


#-------------------------------------------
class LayoutSource(IteratableRootBase):
    # This class is stream originator
    is_root = True


    def __init__(
        self,
        layout,
        load_policy,
        deliverables,
        artifact_store,
        op_params: LayoutOpParams
    ):
        super().__init__()


        self.layout = layout
        self.load_policy = set(load_policy)
        self.artifact_store = artifact_store
        self.op_params = op_params

        self.serializers = self.layout.available_modalities
        # get a copy of all available modalities in layout
        self._capabilities = list(
            self.serializers.keys()
        )
        self._capabilities.extend(
            name for name in self.artifact_store.names()
            if name not in self._capabilities
        )


        contract = self.capabilities if deliverables is None else list(deliverables)
        missing = sorted(set(contract) - set(self.capabilities))
        if missing:
            raise ValueError(
                f"Data source cannot satisfy its required contract: {missing}"
            )
        self._deliverables = list(dict.fromkeys(contract))

    # -------------------------------------------------
    def _decode_sample(self,
                       sample,
                       requires):

        out = {
            "sample_id": sample["__key__"],
            "__shard__": sample["__shard__"],
            "domain": self.layout.domain,
        }

        available = {
            k: v
            for k, v in sample.items()
            if not k.startswith("__")
        }

        for required in requires:

            if self.artifact_store.has(required):
                artifact = self.artifact_store.get(required)
                out[required] = artifact.value()
                continue

            if required not in available:
                continue

            # modality -> serialization
            serialization = (
                self.serializers[required]
            )

            # serialization -> loader
            loader = FILETYPE_LOADERS[serialization]

            # check and load lazy modalities
            payload = available[required]

            if isinstance(payload, DeferredPayload):
                payload = payload.load()

            # deserialize transport payload
            out[required] = loader(payload)

        return out

    @staticmethod
    # -------------------------------------------------
    def _shuffle_buffered(samples, size):

        if size <= 1:
            yield from samples
            return

        buffer = []

        for sample in samples:
            buffer.append(sample)

            if len(buffer) >= size:
                index = random.randrange(len(buffer))
                yield buffer.pop(index)

        random.shuffle(buffer)

        while buffer:
            yield buffer.pop()

    # -------------------------------------------------
    def _shuffle_buffered_by_shard(self, samples, size):

        for _, shard_samples in groupby(
            samples,
            key=lambda sample: sample["__shard__"],
        ):
            yield from self._shuffle_buffered(
                shard_samples,
                size,
            )

    # =====================================
    # DatasetSourceBase Overrides

    # =====================================
    # NEGOTIATION
    # =====================================

   # "What WILL I offer?
    def request(
            self,
            request
    )-> list[str]:
        raise RuntimeError(
            "LayoutSource contracts are fixed at construction and cannot be negotiated."
        )

    #--------------------------------------
    # Execution
    # -------------------------------------

    def __iter__(self):

        if not self._deliverables:
            return

        samples = self.layout.transport.iter_samples(
            deliverables=self._deliverables,
            load_policy=self.load_policy,
            shardshuffle=self.op_params.shardshuffle,
        )

        if self.op_params.shuffle:
            samples = self._shuffle_buffered_by_shard(
                samples,
                self.op_params.shuffle_buffer,
            )

        for sample in samples:
            data = self._decode_sample(
                sample,
                self._deliverables
            )
            yield data
