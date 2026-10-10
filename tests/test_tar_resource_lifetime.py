import gc
import io
import os
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from runweaver_ml.dataset_management import DatasetPrefetchEngine, MultimodalDataset
from runweaver_ml.dataset_management.engine.transport import TarTransport


def write_tar(path, entries):
    path.parent.mkdir(exist_ok=True)
    with tarfile.open(path, "w:gz" if path.name.endswith(".gz") else "w") as archive:
        for name, data in entries:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def open_descriptors(root):
    descriptors = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if str(root) in target:
            descriptors.append(target)
    return descriptors


@unittest.skipUnless(Path("/proc/self/fd").exists(), "Requires Linux descriptor inspection")
class TarResourceLifetimeTest(unittest.TestCase):
    def test_concurrent_deferred_loads_share_reader_safely(self):
        for suffix in (".tar", ".tar.gz"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_tar(root / "text" / ("000" + suffix), [
                    (f"s{i}.text.txt", str(i).encode()) for i in range(20)
                ])
                transport = TarTransport(root)
                samples = list(transport.iter_samples(deliverables=["text"], load_policy=["text"]))
                self.assertEqual(len(open_descriptors(root)), 1)
                with ThreadPoolExecutor(max_workers=4) as executor:
                    values = list(executor.map(lambda sample: sample["text"].load(), samples))
                self.assertEqual(sorted(int(value) for value in values), list(range(20)))
                self.assertEqual(open_descriptors(root), [])

    def test_repeated_success_and_discovery_close_without_collection(self):
        enabled = gc.isenabled()
        gc.disable()
        try:
            for suffix in (".tar", ".tar.gz"):
                with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    write_tar(root / "text" / ("000" + suffix), [
                        ("s1.text.txt", b"one"), ("s2.text.txt", b"two"),
                    ])
                    for _ in range(20):
                        transport = TarTransport(root)
                        self.assertEqual(open_descriptors(root), [])
                        samples = list(transport.iter_samples(deliverables=["text"], load_policy=[]))
                        self.assertEqual(len(samples), 2)
                        self.assertEqual(open_descriptors(root), [])
        finally:
            if enabled:
                gc.enable()

    def test_retained_webdataset_failures_do_not_retain_open_streams(self):
        for suffix in (".tar", ".tar.gz"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_tar(root / "text" / ("000" + suffix), [("s1.text.txt", b"one")])
                (root / "text" / ("001" + suffix)).write_bytes(b"invalid archive")
                transport = TarTransport(root)
                errors = []
                for _ in range(20):
                    try:
                        list(transport.iter_samples(deliverables=["text"], load_policy=[]))
                    except tarfile.ReadError as exc:
                        errors.append(exc)  # Keep args AND original tracebacks alive.
                    else:
                        self.fail("Corrupt TAR unexpectedly succeeded")
                    self.assertEqual(open_descriptors(root), [])
                self.assertEqual(len(errors), 20)
                for error in errors:
                    self.assertFalse(any(isinstance(arg, io.IOBase) for arg in error.args))

    def test_eager_iterator_close_and_synchronization_failure_close_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tar(root / "text" / "000.tar", [("s1.text.txt", b"one"), ("s2.text.txt", b"two")])
            transport = TarTransport(root)
            iterator = transport._iter_source_shard(transport.manifest.fragments[0], "000.tar")
            next(iterator)
            self.assertEqual(len(open_descriptors(root)), 1)
            iterator.close()
            self.assertEqual(open_descriptors(root), [])
            write_tar(root / "other" / "000.tar", [("s3.other.txt", b"missing overlay")])
            transport = TarTransport(root)
            with self.assertRaisesRegex(RuntimeError, "Missing synchronized sample") as caught:
                list(transport.iter_samples(deliverables=["text", "other"], load_policy=["text"]))
            self.assertIsNotNone(caught.exception)
            self.assertEqual(open_descriptors(root), [])

    def test_deferred_payloads_outlive_iterator_and_abandon_independently(self):
        for suffix in (".tar", ".tar.gz"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_tar(root / "text" / ("000" + suffix), [
                    ("s1.text.txt", b"one"), ("s2.text.txt", b"two"), ("s3.text.txt", b"three"),
                ])
                transport = TarTransport(root)
                iterator = transport.iter_samples(deliverables=["text"], load_policy=["text"])
                first, second = next(iterator), next(iterator)
                iterator.close()  # Keep both payloads, discard remaining map entries.
                self.assertEqual(len(open_descriptors(root)), 1)
                first["text"].close()
                self.assertEqual(len(open_descriptors(root)), 1)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    first["text"].load()
                self.assertEqual(second["text"].load(), b"two")
                self.assertTrue(second["text"].loaded)
                self.assertIsNone(second["text"]._loader)
                self.assertEqual(second["text"].load(), b"two")
                self.assertEqual(open_descriptors(root), [])

    def test_failed_load_and_duplicate_index_close_with_tracebacks_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "text" / "000.tar"
            write_tar(path, [("s1.text.txt", b"x" * 100000)])
            transport = TarTransport(root)
            retained = []
            for _ in range(20):
                write_tar(path, [("s1.text.txt", b"x" * 100000)])
                payload = list(transport.iter_samples(deliverables=["text"], load_policy=["text"]))[0]["text"]
                path.write_bytes(b"")
                try:
                    payload.load()
                except tarfile.ReadError as exc:
                    retained.append(exc)
                else:
                    self.fail("Truncated deferred payload unexpectedly loaded")
                self.assertIsNone(payload._loader)
                self.assertFalse(payload.loaded)
                self.assertEqual(open_descriptors(root), [])
            write_tar(path, [("s1.text.txt", b"one"), ("s1.text.txt", b"duplicate")])
            try:
                list(transport.iter_samples(deliverables=["text"], load_policy=["text"]))
            except RuntimeError as exc:
                retained.append(exc)
            self.assertEqual(len(retained), 21)
            self.assertEqual(open_descriptors(root), [])

    def test_abandoned_payloads_and_partial_fragment_failure_release_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tar(root / "text" / "000.tar", [("s1.text.txt", b"one"), ("s2.text.txt", b"two")])
            transport = TarTransport(root)
            iterator = transport.iter_samples(deliverables=["text"], load_policy=["text"])
            sample = next(iterator)
            iterator.close()
            self.assertEqual(len(open_descriptors(root)), 1)
            del sample
            self.assertEqual(open_descriptors(root), [])
            write_tar(root / "other" / "000.tar", [("s1.other.txt", b"valid")])
            transport = TarTransport(root)
            (root / "other" / "000.tar").write_bytes(b"corrupt")
            errors = []
            for _ in range(10):
                try:
                    list(transport.iter_samples(deliverables=["text", "other"], load_policy=["text", "other"]))
                except tarfile.ReadError as exc:
                    errors.append(exc)
                self.assertEqual(open_descriptors(root), [])
            self.assertEqual(len(errors), 10)

    def test_prefetch_retained_failure_and_lazy_early_close(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tar(root / "text" / "000.tar", [(f"s{i}.text.txt", b"one") for i in range(100)])
            dataset = MultimodalDataset(root, num_samples=100, load_policy=["text"])
            iterator = iter(dataset)
            next(iterator)
            iterator.close()
            self.assertEqual(open_descriptors(root), [])
            (root / "text" / "001.tar").write_bytes(b"invalid archive")
            dataset = MultimodalDataset(root, num_samples=101)
            errors = []
            for _ in range(10):
                try:
                    list(dataset)
                except tarfile.ReadError as exc:
                    errors.append(exc)
                self.assertEqual(open_descriptors(root), [])
            self.assertEqual(len(errors), 10)
    def test_failed_payload_does_not_close_reader_needed_by_sibling(self):
        for suffix in (".tar", ".tar.gz"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_tar(root / "text" / ("000" + suffix), [
                    ("s1.text.txt", b"one"), ("s2.text.txt", b"two"),
                ])
                transport = TarTransport(root)
                samples = list(transport.iter_samples(deliverables=["text"], load_policy=["text"]))
                failed, sibling = samples[0]["text"], samples[1]["text"]
                reader = failed._loader.reader
                with patch.object(reader.archive, "extractfile", side_effect=OSError("read failed")):
                    try:
                        failed.load()
                    except OSError as exc:
                        retained = exc
                self.assertIsNone(failed._loader)
                self.assertEqual(len(open_descriptors(root)), 1)
                self.assertEqual(sibling.load(), b"one")
                self.assertEqual(open_descriptors(root), [])
                self.assertEqual(str(retained), "read failed")

    def test_prefetch_deferred_failure_discards_queued_loaders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tar(root / "text" / "000.tar", [(f"s{i}.text.txt", b"one") for i in range(100)])
            transport = TarTransport(root)
            errors = []

            class Pipeline:
                def iter_root(self):
                    return transport.iter_samples(deliverables=["text"], load_policy=["text"])

                def process_sample(self, sample):
                    raise ValueError("provider failed before decoding")

            for _ in range(10):
                try:
                    list(DatasetPrefetchEngine(Pipeline(), queue_size=10, num_workers=2))
                except ValueError as exc:
                    errors.append(exc)
                self.assertEqual(open_descriptors(root), [])
            self.assertEqual(len(errors), 10)


class PrefetchIteratorCleanupTest(unittest.TestCase):
    def test_fetch_iterator_closes_and_workers_stop_on_cancel_or_provider_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                closed = threading.Event()

                class Pipeline:
                    def iter_root(self):
                        try:
                            for index in range(100):
                                yield {"sample_id": index}
                        finally:
                            closed.set()

                    def process_sample(self, sample):
                        if fail:
                            raise ValueError("provider failed")
                        return sample

                engine = DatasetPrefetchEngine(Pipeline(), queue_size=1, num_workers=2, name="cleanup-test")
                iterator = iter(engine)
                if fail:
                    with self.assertRaisesRegex(ValueError, "provider failed"):
                        next(iterator)
                else:
                    next(iterator)
                    iterator.close()
                self.assertTrue(closed.is_set())
                self.assertFalse(any(thread.name.startswith("cleanup-test-") for thread in threading.enumerate()))
