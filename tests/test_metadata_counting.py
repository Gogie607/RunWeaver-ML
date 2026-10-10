import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from runweaver_ml.dataset_management import MultimodalDataset
from runweaver_ml.dataset_management.engine.metadata.dataset_meta_loader import MetadataLoader
from runweaver_ml.dataset_management.engine.transport import DeferredPayload, TarTransport


def write_shard(path, keys, modality="audio_encoding"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz" if path.name.endswith(".gz") else "w") as archive:
        for key in keys:
            # Deliberately not valid NPY: metadata must never deserialize it.
            data = b"not an encoding array"
            member = tarfile.TarInfo(f"{key}.{modality}.npy")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


class MetadataCountingTest(unittest.TestCase):
    def test_missing_metadata_counts_raw_keys_without_eager_read_or_decode(self):
        for suffix in (".tar", ".tar.gz"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_shard(root / "audio_encoding" / ("000" + suffix), ["a", "b"])
                write_shard(root / "audio_encoding" / ("001" + suffix), ["a", "c"])
                with patch.object(TarTransport, "_eager_fragment_map", side_effect=AssertionError("eager count")), \
                     patch.object(DeferredPayload, "load", side_effect=AssertionError("decoded count")):
                    # Counting remains lazy even when normal delivery is eager.
                    dataset = MultimodalDataset(root, requires=["audio_encoding"], load_policy=[])
                self.assertEqual(dataset.num_samples, 4)
                metadata = json.loads((root / "dataset_metadata.json").read_text())
                self.assertEqual(metadata, {"num_samples": 4, "available_modalities": ["audio_encoding"]})
                with patch.object(TarTransport, "iter_samples", side_effect=AssertionError("cached count iterated")):
                    self.assertEqual(MultimodalDataset(root).num_samples, 4)

    def test_count_preserves_authoritative_modality_and_releases_loaders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            readers_closed = []
            payloads = []
            requests = []

            class Loader:
                def __call__(self):
                    raise AssertionError("Counting loaded a payload")

                def close(self):
                    readers_closed.append(True)

            class Transport:
                def iter_samples(self, **options):
                    requests.append(options)
                    for key in ("a", "b", "c"):
                        payload = DeferredPayload(Loader())
                        payloads.append(payload)
                        yield {"__key__": key, "encoding": payload}

            dataset = SimpleNamespace(root=root, transport=Transport(), available_modalities={"encoding": "npy", "sparse": "npy"})
            self.assertEqual(MetadataLoader.generate(dataset).num_samples, 3)
            self.assertEqual(requests, [{"deliverables": ["encoding"], "load_policy": ["encoding"]}])
            self.assertEqual(len(readers_closed), 3)
            self.assertTrue(all(payload._loader is None and not payload.loaded for payload in payloads))

    def test_count_failure_closes_iterator_and_does_not_cache_partial_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = []

            class Loader:
                def __call__(self):
                    raise AssertionError("Counting loaded a payload")

                def close(self):
                    events.append("payload closed")

            class Transport:
                def iter_samples(self, **options):
                    try:
                        yield {"__key__": "a", "encoding": DeferredPayload(Loader())}
                        raise RuntimeError("broken index")
                    finally:
                        events.append("iterator closed")

            dataset = SimpleNamespace(root=root, transport=Transport(), available_modalities={"encoding": "npy"})
            with self.assertRaisesRegex(RuntimeError, "broken index"):
                MetadataLoader.generate(dataset)
            self.assertEqual(events, ["payload closed", "iterator closed"])
            self.assertFalse((root / "dataset_metadata.json").exists())
