"""Minimal native-value transport demonstrating the public extension contract."""

import random

from .transport_base import TransportBase
from .transport_manifest import TransportManifest


class MemoryTransport(TransportBase):
    """Reiterate normalized sample dictionaries without files or serialization.

    Samples require __key__; optional __shard__ values are preserved, never
    manufactured. Modality values are native Python objects. Top-level dictionaries
    are copied per iteration; callers retain ownership of nested payloads.
    """

    TRANSPORT_NAME = "memory"

    def __init__(self, samples):
        self._samples = []
        identities = set()
        modalities = {}
        partitions = []
        for sample in samples:
            row = dict(sample)
            if "__key__" not in row:
                raise ValueError("Memory samples require __key__")
            identity = (row.get("__shard__"), row["__key__"])
            if identity in identities:
                raise ValueError(f"Duplicate memory sample identity: {identity!r}")
            identities.add(identity)
            shard = row.get("__shard__")
            if shard is not None and shard not in partitions:
                partitions.append(shard)
            modalities.update({key: None for key in row if not key.startswith("__")})
            self._samples.append(row)
        self.manifest = TransportManifest(
            shard_names=partitions, available_modalities=modalities
        )
        self.num_samples = len(self._samples)

    def iter_samples(
        self, *, deliverables, load_policy, shardshuffle=False, anchor_modality=None
    ):
        requested = set(deliverables)
        if anchor_modality is not None and anchor_modality not in requested:
            raise ValueError("Anchor modality must be requested")
        rows = [
            row for row in self._samples
            if anchor_modality is None or anchor_modality in row
        ]
        all_partitioned = all(row.get("__shard__") is not None for row in rows)
        if shardshuffle and not all_partitioned:
            raise ValueError("Shard shuffle requires __shard__ on every sample")
        if all_partitioned:
            # Preserve the existing grouping for explicitly partitioned inputs.
            partitions = list(self.manifest.shard_names)
            if shardshuffle:
                random.shuffle(partitions)
            rows = [
                row for partition in partitions for row in rows
                if row["__shard__"] == partition
            ]
        for row in rows:
            yield {
                key: value for key, value in row.items()
                if key in {"__key__", "__shard__"} or key in requested
            }
