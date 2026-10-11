"""Demonstrate automatic DSM provider installation with an unchanged transport.

Run with RunWeaver available:
    python scripts/demo_memory_provider.py
    python scripts/demo_memory_provider.py --word research
    python scripts/demo_memory_provider.py --rng --seed 42

The training session uses the same build_datasets -> create_loader path.
The provider class is application-defined; defining/importing it registers its
provider_name through DSM's existing ProviderBase mechanism.
"""

import argparse
import random

from runweaver_ml.dataset_management import MemoryTransport, build_datasets, create_loader
from runweaver_ml.dataset_management.source import ProviderBase
from runweaver_ml.dataset_management.artifacts import BaseArtifact


class RandomGeneratorArtifact(BaseArtifact):
    """Own one stateful RNG per dataset, constructed by the artifact factory."""

    artifact_name = "demo_random_generator"

    @classmethod
    def from_config(cls, artifact_config, dataset_config):
        return cls(
            artifact_config["name"],
            random.Random(artifact_config["config"]["seed"]),
        )


class RandomIntegerProvider(ProviderBase):
    """Retrieve dataset-owned state rather than accepting an RNG constructor arg."""

    provider_name = "demo_random_integer"
    artifact_requires = ["sample_rng"]
    capabilities = ["random_integer"]

    def process(self, sample, view):
        rng = view.artifacts.item("sample_rng")
        sample["random_integer"] = rng.randint(0, 99)
        return sample


def demonstrate_rng(seed):
    """Reuse the memory -> configured builder -> loader path with active state."""
    config = {
        "batch_size": 5,
        "requires": ["text", "label"],
        "artifacts": [{
            "name": "sample_rng",
            "handler": "demo_random_generator",
            "config": {"seed": seed},
        }],
        "providers": [{"name": "demo_random_integer"}],
        "iteration_policy": {"mix_mode": "per_domain", "exhaust_policy": "finite"},
        "datasets": [{
            "name": "demo",
            "transport": MemoryTransport([
                {"__key__": str(index), "text": f"sample {index}", "label": index}
                for index in range(5)
            ]),
            # One processing worker makes draw-to-sample assignment reproducible.
            "prefetch_workers": 1,
        }],
    }
    dataset = build_datasets(config)
    loader = create_loader(dataset, config["requires"], config["batch_size"])
    for traversal in (1, 2):
        print(f"\nRNG traversal {traversal}: same artifact, continuing state")
        for batch in loader:
            print(batch)


class AppendWordProvider(ProviderBase):
    """Append a configured word to text, independently of the sample transport."""

    provider_name = "demo_append_word"
    requires = ["text"]
    capabilities = ["text"]

    def __init__(self, word):
        super().__init__()
        if not isinstance(word, str) or not word.strip():
            raise ValueError("word must be a non-empty string")
        self.word = word

    def process(self, sample, view):
        sample["text"] = f"{sample['text']} {self.word}"
        return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word", default="RunWeaver")
    parser.add_argument("--rng", action="store_true", help="Demonstrate a stateful RNG artifact")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rng:
        demonstrate_rng(args.seed)
        return

    transport = MemoryTransport([
        {"__key__": "a", "text": "hello", "label": 1},
        {"__key__": "b", "text": "world", "label": 0},
    ])
    base_config = {
        "batch_size": 2,
        "requires": ["text", "label"],
        "iteration_policy": {
            "mix_mode": "per_domain",
            "exhaust_policy": "finite",
        },
        "datasets": [{
            "name": "demo",
            "transport": transport,
            "prefetch_workers": 1,
        }],
    }
    provider_config = {
        "name": "demo_append_word",
        "config": {"word": args.word},
    }

    # Only the provider configuration differs. The same transport supplies fresh
    # sample dictionaries, so modifying text in one pipeline cannot affect another.
    for title, settings in (
        ("Provider configured", {"providers": [provider_config]}),
        ("Provider removed: original text restored", {}),
    ):
        config = {**base_config, **settings}
        dataset = build_datasets(config)
        loader = create_loader(
            dataset,
            required_modalities=config["requires"],
            batch_size=config["batch_size"],
        )
        print(f"\n{title}")
        for batch in loader:
            print(batch)


if __name__ == "__main__":
    main()
