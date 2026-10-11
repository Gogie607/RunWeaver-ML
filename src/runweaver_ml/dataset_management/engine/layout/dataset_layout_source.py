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
    anchor_modality: str | None = None


#-------------------------------------------
class LayoutSource(IteratableRootBase):
    # This class is stream originator
    is_root = True


    def __init__(
        self,
        transport,
        domain,
        load_policy,
        deliverables,
        artifact_store,
        op_params: LayoutOpParams
    ):
        super().__init__()


        self.transport = transport
        self.domain = domain
        self.load_policy = set(load_policy)
        self.artifact_store = artifact_store
        self.op_params = op_params

        self.serializers = dict(self.transport.manifest.available_modalities)
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

        if (
            self.op_params.anchor_modality is not None
            and self.op_params.anchor_modality not in self._deliverables
        ):
            raise ValueError(
                "Layout anchor modality must be part of the required "
                f"contract: '{self.op_params.anchor_modality}'."
            )

    # -------------------------------------------------
    def _decode_sample(self,
                       sample,
                       requires):

        if "__key__" not in sample:
            raise ValueError("Transport samples require __key__")
        out = {
            "sample_id": sample["__key__"],
            "domain": self.domain,
        }
        if "__shard__" in sample:
            out["__shard__"] = sample["__shard__"]

        available = {
            k: v
            for k, v in sample.items()
            if not k.startswith("__")
        }

        for required in requires:

            if self.artifact_store.has(required):
                out[required] = self.artifact_store.get(required)
                continue

            if required not in available:
                continue

            # modality -> serialization
            serialization = (
                self.serializers[required]
            )

            # serialization -> loader
            loader = FILETYPE_LOADERS.get(serialization)
            if serialization is not None and loader is None:
                raise ValueError(
                    f"Unknown serialization {serialization!r} for modality {required!r}"
                )

            # check and load lazy modalities
            payload = available[required]

            if isinstance(payload, DeferredPayload):
                payload = payload.load()

            # deserialize transport payload
            out[required] = payload if serialization is None else loader(payload)

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

        def shard_key(sample):
            if sample.get("__shard__") is None:
                raise ValueError("Partition-based shuffle requires __shard__ on every sample")
            return sample["__shard__"]

        for _, shard_samples in groupby(samples, key=shard_key):
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

        samples = self.transport.iter_samples(
            deliverables=self._deliverables,
            load_policy=self.load_policy,
            shardshuffle=self.op_params.shardshuffle,
            anchor_modality=self.op_params.anchor_modality,
        )

        if self.op_params.shuffle:
            samples = self._shuffle_buffered_by_shard(
                samples,
                self.op_params.shuffle_buffer,
            )

        for sample in samples:
            if self.op_params.shardshuffle and sample.get("__shard__") is None:
                raise ValueError("Shard shuffle requires __shard__ on every sample")
            data = self._decode_sample(
                sample,
                self._deliverables
            )
            yield data
