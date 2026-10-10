"""Demonstrate automatic DSM provider installation with an unchanged transport.

Run with RunWeaver available:
    python scripts/demo_memory_provider.py
    python scripts/demo_memory_provider.py --word research

The training session uses the same build_datasets -> create_loader path.
The provider class is application-defined; defining/importing it registers its
provider_name through DSM's existing ProviderBase mechanism.
"""

import argparse

from runweaver_ml.dataset_management import MemoryTransport, build_datasets, create_loader
from runweaver_ml.dataset_management.source import ProviderBase


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
    args = parser.parse_args()

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
