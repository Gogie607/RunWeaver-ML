from pathlib import Path
from ..transport.transport_base import TransportBase


class DatasetLayout:

    def __init__(
        self,
        domain,
        root,
        transport: TransportBase,
    ):

        self.domain = domain
        self.root = Path(root)
        self.transport = transport
        self.manifest = transport.manifest

        # TODO: Local root validation remains here for Stage 1 compatibility.
        # Move it into concrete transports when CSV construction is added.
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist: {self.root}"
            )

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"Dataset root must be a local directory: {self.root}"
            )

        self.available_modalities = dict(
            self.manifest.available_modalities
        )

    def summary(self):

        modalities = ", ".join(
            self.available_modalities.keys()
        )

        return (
            "\nDataset Summary\n"
            "---------------\n"
            f"Domain: {self.domain}\n"
            f"Transport: {self.transport.TRANSPORT_NAME}\n"
            f"Shards: {len(self.manifest.shard_names)}\n"
            f"Modalities: {modalities}\n"
            #f"Samples: {self.num_samples}\n"
        )
   # -------------------------------------------------

    def __repr__(self):
        return self.summary()
