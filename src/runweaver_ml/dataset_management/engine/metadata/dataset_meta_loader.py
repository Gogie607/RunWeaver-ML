
import json

from .dataset_metadata import DatasetMetadata


class MetadataLoader:

    BASENAME = "dataset_metadata"

    # -------------------------------------
    @classmethod
    def load(cls, dataset):

        path = dataset.root / f"{cls.BASENAME}.json"

        try:
            with open(path, "r") as f:
                data = json.load(f)

        except FileNotFoundError:
            return None

        try:
            return DatasetMetadata(
                num_samples=data["num_samples"],
                available_modalities=data["available_modalities"],
            )

        except KeyError:
            return None

    # -------------------------------------

    @classmethod
    def save(
            cls,
            dataset,
            meta,
    ):

        path = dataset.root / f"{cls.BASENAME}.json"

        with open(path, "w") as f:

            json.dump(
                {
                    "num_samples": meta.num_samples,
                    "available_modalities":
                        meta.available_modalities,
                },
                f,
                indent=2,
            )

    # -------------------------------------

    @classmethod
    def generate(
            cls,
            dataset,
    ):

        #
        # authoritative modality
        #
        modality = next(
            iter(dataset.layout.available_modalities)
        )

        view = dataset.create_view(required=[modality])

        count = 0

        for _ in view:
            count += 1

        meta = DatasetMetadata(
            num_samples=count,
            available_modalities=list(
                dataset.layout.available_modalities
            )
        )

        cls.save(
            dataset,
            meta
        )

        return meta

    # -------------------------------------

    @classmethod
    def load_or_generate(
            cls,
            dataset,
    ):

        meta = cls.load(dataset)

        if meta is not None:
            return meta

        return cls.generate(dataset)
