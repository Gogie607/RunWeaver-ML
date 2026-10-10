import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from runweaver_ml.dataset_management import (
    DeferredPayload, MemoryTransport, MultimodalDataset, TransportBase,
    TransportManifest, DynamicCollator, build_datasets, create_loader,
)
from runweaver_ml.dataset_management.artifacts import Artifact
from runweaver_ml.dataset_management.source import ProviderBase


class UppercaseProvider(ProviderBase):
    requires = ["text"]
    artifact_requires = ["suffix"]
    capabilities = ["upper"]

    def process(self, sample, view):
        if sample["text"] == "skip":
            return None
        sample["upper"] = sample["text"].upper() + view.artifacts.get("suffix").value()
        return sample


class ExternalTransport(TransportBase):
    """Test-local contributor implementation using only the public contract."""

    TRANSPORT_NAME = "external"

    def __init__(self):
        self.manifest = TransportManifest(available_modalities={"text": "txt", "n": None})
        self.loaded = 0

    def _load_text(self):
        self.loaded += 1
        return b"hello"

    def iter_samples(self, *, deliverables, load_policy, shardshuffle=False,
                     anchor_modality=None):
        if shardshuffle or anchor_modality is not None:
            raise ValueError("This test transport does not support partition/anchor options")
        sample = {"__key__": 17}
        if "text" in deliverables:
            sample["text"] = DeferredPayload(self._load_text)
        if "n" in deliverables:
            sample["n"] = 3
        yield sample


class ShardWriterProvider(ProviderBase):
    """Representative application persistence stage; no framework writer exists."""

    requires = ["text", "__shard__"]
    capabilities = ["generated"]

    def __init__(self, destination):
        super().__init__()
        self.destination = Path(destination)

    def process(self, sample, view):
        value = sample["text"].upper().encode("utf-8")
        with tarfile.open(self.destination / sample["__shard__"], "w") as archive:
            info = tarfile.TarInfo(f"{sample['sample_id']}.generated.txt")
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
        sample["generated"] = value.decode("utf-8")
        return sample


class TransportExtensionTest(unittest.TestCase):
    def test_memory_end_to_end_with_provider_filter_artifact_and_reiteration(self):
        dataset = MultimodalDataset(
            transport=MemoryTransport([
                {"__key__": 1, "text": "hello", "vector": np.array([1, 2])},
                {"__key__": 2, "text": "skip", "vector": np.array([3, 4])},
            ]),
            domain="native", artifacts=[Artifact("suffix", "!")],
            prefetch_workers=2,
        )
        dataset.install_provider(UppercaseProvider())
        self.assertIsNone(dataset.root)
        self.assertEqual(dataset.num_samples, 2)
        loader = create_loader(dataset, [], 2)
        for _ in range(2):
            batches = list(loader)
            self.assertEqual(len(batches), 1)
            self.assertEqual(batches[0]["sample_id"], [1])
            self.assertNotIn("__shard__", batches[0])
            self.assertEqual(batches[0]["domain"], ["native"])
            self.assertEqual(batches[0]["upper"], ["HELLO!"])
            self.assertEqual(batches[0]["vector"].tolist(), [[1., 2.]])

    def test_external_transport_count_does_not_load_and_pipeline_decodes(self):
        transport = ExternalTransport()
        dataset = MultimodalDataset(transport=transport, domain="external", load_policy=["text"])
        self.assertEqual(dataset.num_samples, 1)
        self.assertEqual(transport.loaded, 0)
        batch = next(iter(create_loader(dataset, [], 1)))
        self.assertNotIn("__shard__", batch)
        self.assertEqual(batch["text"], ["hello"])
        self.assertEqual(batch["n"].tolist(), [3])
        self.assertEqual(transport.loaded, 1)

    def test_builder_and_manager_mix_memory_with_tar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "text").mkdir()
            with tarfile.open(root / "text" / "000.tar", "w") as archive:
                info = tarfile.TarInfo("tar1.text.txt")
                info.size = 3
                archive.addfile(info, io.BytesIO(b"tar"))
            manager = build_datasets({
                "requires": ["text"],
                "iteration_policy": {"mix_mode": "mixed"},
                "datasets": [
                    {"name": "tar", "root": str(root)},
                    {"name": "memory", "transport": MemoryTransport([
                        {"__key__": 5, "text": "native"}
                    ])},
                ],
            })
            batches = list(create_loader(manager, [], 2))
            self.assertEqual(sorted(batches[0]["text"]), ["native", "tar"])
            by_domain = dict(zip(batches[0]["domain"], batches[0]["__shard__"]))
            self.assertEqual(by_domain, {"tar": "000.tar", "memory": None})
            self.assertTrue((root / "dataset_metadata.json").is_file())

    def test_native_none_and_deferred_none(self):
        payload = DeferredPayload(lambda: None)
        self.assertIsNone(payload.load())
        self.assertIsNone(payload.load())
        self.assertTrue(payload.loaded)
        dataset = MultimodalDataset(
            transport=MemoryTransport([{"__key__": 1, "value": payload}]), domain="nullable"
        )
        self.assertIsNone(list(dataset)[0]["value"])

    def test_required_fields_and_unknown_serializers_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "required contract"):
            MultimodalDataset(transport=MemoryTransport([]), domain="empty", requires=["absent"])
        transport = ExternalTransport()
        transport.manifest.available_modalities["text"] = "unknown"
        dataset = MultimodalDataset(transport=transport, domain="bad")
        with self.assertRaisesRegex(ValueError, "Unknown serialization"):
            list(dataset)

    def test_explicit_domain_count_and_default_root_validation(self):
        with self.assertRaisesRegex(ValueError, "explicit domain"):
            MultimodalDataset(transport=MemoryTransport([]))
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            MultimodalDataset(transport=MemoryTransport([]), domain="bad", num_samples=-1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                MultimodalDataset(Path(directory) / "missing")
            file = Path(directory) / "file"
            file.write_text("x")
            with self.assertRaises(NotADirectoryError):
                MultimodalDataset(file)

    def test_anchor_projection_and_partition_shuffle(self):
        transport = MemoryTransport([
            {"__key__": 1, "text": "one", "__shard__": "a"},
            {"__key__": 2, "text": "two", "response": "yes", "__shard__": "b"},
        ])
        dataset = MultimodalDataset(
            transport=transport, domain="anchor", requires=["response"],
            op_params={"layout": {"anchor_modality": "response", "shuffle": True,
                                   "shardshuffle": True, "shuffle_buffer": 2}},
        )
        self.assertEqual(dataset.num_samples, 1)
        sample = list(dataset)[0]
        self.assertEqual(sample["sample_id"], 2)
        self.assertNotIn("text", sample)

    def test_memory_does_not_manufacture_partitions(self):
        transport = MemoryTransport([{"__key__": 1, "text": "native"}])
        self.assertEqual(transport.manifest.shard_names, [])
        raw = list(transport.iter_samples(deliverables=["text"], load_policy=[]))
        self.assertEqual(raw, [{"__key__": 1, "text": "native"}])
        partitioned = MemoryTransport([
            {"__key__": 1, "__shard__": "a", "text": "one"},
            {"__key__": 2, "__shard__": "b", "text": "two"},
            {"__key__": 3, "__shard__": "a", "text": "three"},
        ])
        rows = list(partitioned.iter_samples(deliverables=["text"], load_policy=[]))
        self.assertEqual([row["__key__"] for row in rows], [1, 3, 2])
        with self.assertRaisesRegex(ValueError, "require __key__"):
            MemoryTransport([{"text": "missing identity"}])

    def test_collator_preserves_only_supplied_shard_identity(self):
        collate = DynamicCollator(["text", "__shard__"])
        shardless = {"sample_id": 1, "domain": "plain", "text": "one"}
        sharded = {"sample_id": 2, "domain": "tar", "text": "two",
                   "__shard__": "000.tar"}
        self.assertNotIn("__shard__", collate([shardless]))
        self.assertEqual(collate([sharded])["__shard__"], ["000.tar"])
        self.assertEqual(collate([shardless, sharded])["__shard__"], [None, "000.tar"])

    def test_external_samples_still_require_key(self):
        transport = ExternalTransport()
        transport.iter_samples = lambda **kwargs: iter([{"text": b"hello"}])
        dataset = MultimodalDataset(transport=transport, domain="bad", num_samples=1)
        with self.assertRaisesRegex(ValueError, "require __key__"):
            list(create_loader(dataset, [], 1))

    def test_partition_operations_explicitly_require_shards(self):
        for options in ({"shuffle": True}, {"shardshuffle": True}):
            with self.subTest(options=options):
                dataset = MultimodalDataset(
                    transport=MemoryTransport([{"__key__": 1, "text": "one"}]),
                    domain="plain", op_params={"layout": options},
                )
                with self.assertRaisesRegex(ValueError, "requires __shard__"):
                    list(create_loader(dataset, [], 1))

        # Even a contributor transport that silently ignores shardshuffle is checked.
        transport = ExternalTransport()
        transport.iter_samples = lambda **kwargs: iter([{"__key__": 1, "text": b"one"}])
        dataset = MultimodalDataset(
            transport=transport, domain="external", num_samples=1,
            op_params={"layout": {"shardshuffle": True}},
        )
        with self.assertRaisesRegex(ValueError, "requires __shard__"):
            list(dataset)

    def test_shard_aware_provider_rejects_shardless_samples_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = MultimodalDataset(
                transport=MemoryTransport([{"__key__": 1, "text": "one"}]),
                domain="plain",
            )
            dataset.install_provider(ShardWriterProvider(directory))
            with self.assertRaisesRegex(ValueError, "ShardWriterProvider requires __shard__"):
                list(create_loader(dataset, [], 1))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_tar_shard_identity_reaches_persistent_provider_and_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "text").mkdir()
            destination = root / "generated"
            # Keep provider outputs outside transport discovery during construction.
            with tarfile.open(root / "text" / "000.tar", "w") as archive:
                info = tarfile.TarInfo("tar1.text.txt")
                info.size = 3
                archive.addfile(info, io.BytesIO(b"tar"))
            dataset = MultimodalDataset(root, requires=["text"])
            destination.mkdir()
            dataset.install_provider(ShardWriterProvider(destination))
            batch = next(iter(create_loader(dataset, [], 1)))
            self.assertEqual(batch["__shard__"], ["000.tar"])
            self.assertEqual(batch["generated"], ["TAR"])
            with tarfile.open(destination / "000.tar") as archive:
                self.assertEqual(archive.extractfile("tar1.generated.txt").read(), b"TAR")


if __name__ == "__main__":
    unittest.main()
