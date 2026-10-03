import io
import tarfile
import unittest
from pathlib import Path

import numpy as np

from runweaver_ml.dataset_management.artifacts import ArtifactStore
from runweaver_ml.dataset_management.engine.layout.dataset_layout import DatasetLayout
from runweaver_ml.dataset_management.engine.layout.dataset_layout_source import (
    LayoutOpParams,
    LayoutSource,
)
from runweaver_ml.dataset_management.engine.transport.tar_transport import (
    TarTransport,
)
from runweaver_ml.dataset_management.engine.transport.transport_base import (
    DeferredPayload,
)
from runweaver_ml.dataset_management.views.dataset_view import DatasetView


def _txt_bytes(value):
    return value.encode("utf-8")


def _npy_bytes(value):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value, dtype=np.float32))
    return buffer.getvalue()


def _write_tar(path, samples):
    with tarfile.open(path, "w") as tar:
        for sample_id, payloads in samples.items():
            for member_suffix, data in payloads.items():
                encoded = data() if callable(data) else data
                info = tarfile.TarInfo(f"{sample_id}.{member_suffix}")
                info.size = len(encoded)
                tar.addfile(info, io.BytesIO(encoded))


def _build_layout(root):
    transport = TarTransport(root)
    return DatasetLayout(
        domain="test-domain",
        root=root,
        transport=transport,
    )


def _build_source(layout, required, load_policy=(), op_params=None):
    return LayoutSource(
        layout=layout,
        load_policy=load_policy,
        deliverables=required,
        artifact_store=ArtifactStore(),
        op_params=op_params or LayoutOpParams(),
    )


def _build_layout_with_manifest_only():

    class Layout:

        domain = "test-domain"
        available_modalities = {}

    return Layout()


class TarTransportLayoutBoundaryTest(unittest.TestCase):

    def test_manifest_publishes_normalized_modalities(self):
        with self.tmp_dir() as root:
            text_dir = root / "text"
            vector_dir = root / "vector"
            text_dir.mkdir()
            vector_dir.mkdir()

            _write_tar(text_dir / "000.tar", {"s1": {"text.txt": _txt_bytes("hello")}})
            _write_tar(vector_dir / "000.tar", {"s1": {"embedding.npy": _npy_bytes([1, 2])}})

            layout = _build_layout(root)

            self.assertEqual(
                layout.available_modalities,
                {
                    "text": "txt",
                    "embedding": "npy",
                },
            )
            self.assertEqual(
                layout.transport.manifest.available_modalities,
                layout.available_modalities,
            )

    def test_one_fragment_dataset_yields_decoded_view_samples(self):
        with self.tmp_dir() as root:
            text_dir = root / "text"
            text_dir.mkdir()
            _write_tar(
                text_dir / "000.tar",
                {
                    "s1": {"text.txt": _txt_bytes("hello")},
                    "s2": {"text.txt": _txt_bytes("world")},
                },
            )

            view = DatasetView(_build_source(_build_layout(root), ["text"]))
            samples = list(view)

            by_id = {sample["sample_id"]: sample for sample in samples}
            self.assertEqual(set(by_id), {"s1", "s2"})
            self.assertEqual(
                by_id["s1"],
                {
                    "sample_id": "s1",
                    "__shard__": "000.tar",
                    "domain": "test-domain",
                    "text": "hello",
                },
            )
            self.assertEqual(
                by_id["s2"],
                {
                    "sample_id": "s2",
                    "__shard__": "000.tar",
                    "domain": "test-domain",
                    "text": "world",
                },
            )

    def test_multi_fragment_dataset_synchronizes_by_sample_key(self):
        with self.tmp_dir() as root:
            text_dir = root / "text"
            vector_dir = root / "vector"
            text_dir.mkdir()
            vector_dir.mkdir()

            _write_tar(
                text_dir / "000.tar",
                {
                    "s1": {"text.txt": _txt_bytes("hello")},
                    "s2": {"text.txt": _txt_bytes("world")},
                },
            )
            _write_tar(
                vector_dir / "000.tar",
                {
                    "s1": {"embedding.npy": _npy_bytes([1, 2])},
                    "s2": {"embedding.npy": _npy_bytes([3, 4])},
                },
            )

            samples = list(
                DatasetView(_build_source(_build_layout(root), ["text", "embedding"]))
            )

            by_id = {sample["sample_id"]: sample for sample in samples}
            self.assertEqual(by_id["s1"]["text"], "hello")
            np.testing.assert_array_equal(
                by_id["s1"]["embedding"],
                np.asarray([1, 2], dtype=np.float32),
            )
            self.assertEqual(by_id["s2"]["text"], "world")
            np.testing.assert_array_equal(
                by_id["s2"]["embedding"],
                np.asarray([3, 4], dtype=np.float32),
            )

    def test_missing_synchronized_key_still_raises(self):
        with self.tmp_dir() as root:
            text_dir = root / "text"
            vector_dir = root / "vector"
            text_dir.mkdir()
            vector_dir.mkdir()

            _write_tar(text_dir / "000.tar", {"s1": {"text.txt": _txt_bytes("hello")}})
            _write_tar(vector_dir / "000.tar", {"s2": {"embedding.npy": _npy_bytes([1])}})

            view = DatasetView(_build_source(_build_layout(root), ["text", "embedding"]))

            with self.assertRaisesRegex(RuntimeError, "Missing synchronized sample"):
                list(view)

    def test_explicit_anchor_allows_sparse_modality_overlay(self):
        with self.tmp_dir() as root:
            full_dir = root / "a_full"
            sparse_dir = root / "z_sparse"
            full_dir.mkdir()
            sparse_dir.mkdir()

            _write_tar(
                full_dir / "000.tar",
                {
                    "s1": {"text.txt": _txt_bytes("one")},
                    "s2": {"text.txt": _txt_bytes("two")},
                },
            )
            _write_tar(
                sparse_dir / "000.tar",
                {"s2": {"response.txt": _txt_bytes("answer")}},
            )

            source = _build_source(
                _build_layout(root),
                ["text", "response"],
                op_params=LayoutOpParams(anchor_modality="response"),
            )

            samples = list(DatasetView(source))

            self.assertEqual(
                samples,
                [
                    {
                        "sample_id": "s2",
                        "__shard__": "000.tar",
                        "domain": "test-domain",
                        "text": "two",
                        "response": "answer",
                    }
                ],
            )

    def test_default_anchor_preserves_manifest_order(self):
        with self.tmp_dir() as root:
            full_dir = root / "a_full"
            sparse_dir = root / "z_sparse"
            full_dir.mkdir()
            sparse_dir.mkdir()

            _write_tar(
                full_dir / "000.tar",
                {
                    "s1": {"text.txt": _txt_bytes("one")},
                    "s2": {"text.txt": _txt_bytes("two")},
                },
            )
            _write_tar(
                sparse_dir / "000.tar",
                {"s2": {"response.txt": _txt_bytes("answer")}},
            )

            view = DatasetView(
                _build_source(
                    _build_layout(root),
                    ["text", "response"],
                )
            )

            with self.assertRaisesRegex(RuntimeError, "Missing synchronized sample"):
                list(view)

    def test_anchor_modality_must_be_requested(self):
        with self.tmp_dir() as root:
            text_dir = root / "text"
            text_dir.mkdir()
            _write_tar(
                text_dir / "000.tar",
                {"s1": {"text.txt": _txt_bytes("hello")}},
            )

            with self.assertRaisesRegex(
                ValueError,
                "anchor modality must be part of the required contract",
            ):
                _build_source(
                    _build_layout(root),
                    ["text"],
                    op_params=LayoutOpParams(anchor_modality="response"),
                )

    def test_requested_deliverables_restrict_loaded_modalities(self):
        with self.tmp_dir() as root:
            text_dir = root / "text"
            vector_dir = root / "vector"
            text_dir.mkdir()
            vector_dir.mkdir()

            _write_tar(text_dir / "000.tar", {"s1": {"text.txt": _txt_bytes("hello")}})
            _write_tar(vector_dir / "000.tar", {"s1": {"embedding.npy": _npy_bytes([1])}})

            samples = list(DatasetView(_build_source(_build_layout(root), ["text"])))

            self.assertEqual(
                samples,
                [
                    {
                        "sample_id": "s1",
                        "__shard__": "000.tar",
                        "domain": "test-domain",
                        "text": "hello",
                    }
                ],
            )

    def test_lazy_loading_still_uses_deferred_payloads(self):
        with self.tmp_dir() as root:
            vector_dir = root / "vector"
            vector_dir.mkdir()
            _write_tar(vector_dir / "000.tar", {"s1": {"embedding.npy": _npy_bytes([5, 6])}})

            layout = _build_layout(root)
            raw_sample = next(
                layout.transport.iter_samples(
                    deliverables=["embedding"],
                    load_policy=["embedding"],
                )
            )

            self.assertIsInstance(raw_sample["embedding"], DeferredPayload)
            self.assertFalse(raw_sample["embedding"].loaded)

            decoded = list(
                DatasetView(
                    _build_source(
                        layout,
                        ["embedding"],
                        load_policy=["embedding"],
                    )
                )
            )

            np.testing.assert_array_equal(
                decoded[0]["embedding"],
                np.asarray([5, 6], dtype=np.float32),
            )

    def test_shuffle_options_continue_to_iterate(self):
        with self.tmp_dir() as root:
            text_dir = root / "text"
            text_dir.mkdir()
            _write_tar(
                text_dir / "000.tar",
                {
                    "s1": {"text.txt": _txt_bytes("one")},
                    "s2": {"text.txt": _txt_bytes("two")},
                    "s3": {"text.txt": _txt_bytes("three")},
                    "s4": {"text.txt": _txt_bytes("four")},
                },
            )
            _write_tar(text_dir / "001.tar", {"s2": {"text.txt": _txt_bytes("two")}})

            source = _build_source(
                _build_layout(root),
                ["text"],
                op_params=LayoutOpParams(
                    shuffle=True,
                    shardshuffle=True,
                    shuffle_buffer=2,
                ),
            )

            samples = list(DatasetView(source))

            self.assertEqual(
                {sample["sample_id"] for sample in samples},
                {"s1", "s2", "s3", "s4"},
            )
            self.assertEqual(
                {sample["__shard__"] for sample in samples},
                {"000.tar", "001.tar"},
            )

    def test_per_shard_shuffle_starts_before_complete_shard_is_consumed(self):
        consumed = 0

        def samples():
            nonlocal consumed

            for index in range(5):
                consumed += 1
                yield {
                    "__key__": f"s{index}",
                    "__shard__": "000.tar",
                }

        source = _build_source(
            _build_layout_with_manifest_only(),
            [],
            op_params=LayoutOpParams(
                shuffle=True,
                shuffle_buffer=2,
            ),
        )

        shuffled = source._shuffle_buffered_by_shard(
            samples(),
            size=2,
        )

        next(shuffled)

        self.assertEqual(consumed, 2)

    @staticmethod
    def tmp_dir():
        import tempfile

        class TempPath:

            def __enter__(self):
                self._temp_dir = tempfile.TemporaryDirectory()
                return Path(self._temp_dir.__enter__())

            def __exit__(self, exc_type, exc, traceback):
                return self._temp_dir.__exit__(exc_type, exc, traceback)

        return TempPath()


if __name__ == "__main__":
    unittest.main()
