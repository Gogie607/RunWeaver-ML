from dataclasses import dataclass


@dataclass
class DatasetMetadata:

    num_samples: int

    available_modalities: list[str]
