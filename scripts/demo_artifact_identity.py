"""Compare evaluated values, independent RNGs, and shared RNG state in DSM.

Run with RunWeaver available:
    python scripts/demo_artifact_identity.py

Five completed traversals of a one-sample source give five draws per domain.
No iterator is left partly consumed, so prefetch cannot hide extra RNG draws.
"""

import random

from runweaver_ml.dataset_management import MemoryTransport, build_datasets, create_loader
from runweaver_ml.dataset_management.source import ProviderBase, create_providers


DOMAINS = ("alpha", "beta", "gamma")
SEEDS = dict(zip(DOMAINS, (42, 43, 44)))


class ArtifactIntegerProvider(ProviderBase):
    """Interpret the artifact as an evaluated integer or an active RNG.

    Recorded observations are demonstration instrumentation, not RNG ownership.
    """

    provider_name = "demo_artifact_integer"
    artifact_requires = ["sample_rng"]
    capabilities = ["random_integer"]

    def __init__(self):
        super().__init__()
        self.observed = []

    def process(self, sample, view):
        content = view.artifacts.item("sample_rng")
        value = content.randint(0, 99) if isinstance(content, random.Random) else content
        sample["random_integer"] = value
        self.observed.append(value)
        return sample


def demonstrate(case, distribution):
    # The same transport object is used by three independently built datasets.
    source = MemoryTransport([{"__key__": "a", "text": "hello"}])
    manager = build_datasets({
        "requires": ["text"],
        "iteration_policy": {"mix_mode": "per_domain", "exhaust_policy": "finite"},
        "datasets": [
            {"name": domain, "transport": source, "prefetch_workers": 1}
            for domain in DOMAINS
        ],
    })
    manager.distribute_artifact("sample_rng", **distribution)

    # distribute_artifact operates on existing datasets. Install the dependent
    # providers afterward: DSM rejects missing artifacts during installation.
    provider_config = [{"name": "demo_artifact_integer"}]
    providers = []
    loaders = []
    stores = [dataset.artifact_store for dataset in manager.datasets]
    contents = [store.item("sample_rng") for store in stores]
    assert len({id(store) for store in stores}) == 3
    if case == "B":
        assert len({id(content) for content in contents}) == 3
    else:
        assert all(content is contents[0] for content in contents)

    print(f"\nCase {case}: identities (addresses vary between executions)")
    for dataset, store in zip(manager.datasets, stores):
        print(f"{dataset.domain}: store id={id(store)}, content id={id(store.item('sample_rng'))}")
        provider = create_providers(provider_config)[0]
        dataset.install_provider(provider)
        providers.append(provider)
        loaders.append(create_loader(dataset, ["text"], batch_size=1))

    for traversal in range(1, 6):
        print(f"Traversal {traversal}: alpha -> beta -> gamma")
        for index, loader in enumerate(loaders):
            before = [content.getstate() for content in contents] if case != "A" else None
            for batch in loader:
                print(batch)
            if case == "B":
                assert contents[index].getstate() != before[index]
                assert all(contents[other].getstate() == before[other]
                           for other in range(3) if other != index)
            elif case == "C":
                assert all(content.getstate() != before[other]
                           for other, content in enumerate(contents))

    observed = {domain: provider.observed for domain, provider in zip(DOMAINS, providers)}
    if case == "A":
        expected = {domain: [distribution["value"]] * 5 for domain in DOMAINS}
    elif case == "B":
        expected = {}
        for domain in DOMAINS:
            oracle = random.Random(SEEDS[domain])
            expected[domain] = [oracle.randint(0, 99) for _ in range(5)]
    else:
        oracle = random.Random(42)
        sequence = [oracle.randint(0, 99) for _ in range(15)]
        expected = {domain: sequence[index::3] for index, domain in enumerate(DOMAINS)}
        print("Shared sequence in consumption order:", sequence)
    assert observed == expected, (observed, expected)
    print("Observed per-domain values:", observed)
    print("Prediction verified; store/resource identity and state checks passed.")


def main():
    evaluated = random.Random(42).randint(0, 99)
    configurations = (
        ("A", {"value": evaluated}),
        ("B", {"create": lambda dataset: random.Random(SEEDS[dataset.domain])}),
        ("C", {"value": random.Random(42)}),
    )
    for case, distribution in configurations:
        demonstrate(case, distribution)

    # Equal initial sequences alone do not establish shared identity or state.
    left, right = random.Random(42), random.Random(42)
    assert left is not right and left.getstate() == right.getstate()
    assert left.randint(0, 99) == right.randint(0, 99)
    untouched = right.getstate()
    left.randint(0, 99)
    assert right.getstate() == untouched
    print("\nEqual-seed control: distinct RNGs can produce equal values while retaining independent state.")


if __name__ == "__main__":
    main()
