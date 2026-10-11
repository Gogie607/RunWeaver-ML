"""Regression coverage for artifact-backed layout field selection."""

import unittest
from unittest.mock import Mock, call

from runweaver_ml.dataset_management import MemoryTransport, MultimodalDataset
from runweaver_ml.dataset_management.source import ProviderBase
from runweaver_ml.dataset_management.engine.layout.dataset_layout_source import LayoutSource, LayoutOpParams


class ArtifactLayoutContractTest(unittest.TestCase):
    def test_provider_dependencies_check_presence_without_evaluation_or_selection(self):
        class Consumer(ProviderBase):
            artifact_requires = ["resource"]

            def process(self, sample, view):
                return sample

        source = Mock(side_effect=AssertionError("resource evaluated"))
        dataset = self.make_dataset(requires=["text"], artifacts={"resource": source})
        dataset.install_provider(Consumer())
        self.assertEqual(dataset.view.root.deliverables, ["text"])
        self.assertNotIn("resource", list(dataset.view)[0])
        source.assert_not_called()
        missing = self.make_dataset(requires=["text"], artifacts=[])
        with self.assertRaisesRegex(RuntimeError, "missing artifacts"):
            missing.install_provider(Consumer())

    def test_existing_default_selection_paths(self):
        dataset = self.make_dataset(requires=None, artifacts={"language_id": 7})
        self.assertEqual(dataset.view.root.deliverables, ["text"])
        self.assertNotIn("language_id", list(dataset.view.iter_root())[0])
        root = LayoutSource(dataset.transport, "demo", [], None,
                            dataset.artifact_store, LayoutOpParams())
        self.assertEqual(list(root)[0]["language_id"], 7)

    def make_dataset(self, *, requires, artifacts, samples=None):
        return MultimodalDataset(
            transport=MemoryTransport(samples if samples is not None else [
                {"__key__": "a", "text": "hello"},
                {"__key__": "b", "text": "world"},
            ]),
            domain="demo",
            requires=requires,
            artifacts=artifacts,
        )

    def test_requires_selects_transport_and_artifact_fields(self):
        dataset = self.make_dataset(
            requires=["text", "language_id"],
            artifacts={"language_id": 7},
        )
        self.assertEqual(list(dataset.view.iter_root()), [
            {"sample_id": "a", "domain": "demo", "text": "hello", "language_id": 7},
            {"sample_id": "b", "domain": "demo", "text": "world", "language_id": 7},
        ])

    def test_selected_callable_artifact_resolves_without_providers(self):
        source = Mock(side_effect=[10, 11])
        dataset = self.make_dataset(
            requires=["text", "generated"],
            artifacts={"generated": source},
        )
        self.assertEqual(dataset.view.pipeline, [])
        rows = list(dataset.view.iter_root())
        self.assertEqual([row["generated"] for row in rows], [10, 11])
        self.assertEqual(source.call_args_list, [call(), call()])

    def test_unselected_artifacts_are_not_inserted_or_evaluated(self):
        source = Mock(side_effect=AssertionError("unselected artifact evaluated"))
        dataset = self.make_dataset(
            requires=["text"],
            artifacts={"unused": source},
        )
        rows = list(dataset.view.iter_root())
        self.assertTrue(all(set(row) == {"sample_id", "domain", "text"} for row in rows))
        source.assert_not_called()

    def test_selected_artifact_takes_precedence_over_transport_field(self):
        dataset = self.make_dataset(
            requires=["text", "language_id"],
            artifacts={"language_id": 7},
            samples=[{"__key__": "a", "text": "hello", "language_id": 99}],
        )
        self.assertEqual(list(dataset.view.iter_root())[0]["language_id"], 7)


if __name__ == "__main__":
    unittest.main()
