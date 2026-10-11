"""Direct source resolution and resource identity contracts."""
import random
import unittest
from unittest.mock import Mock, call

from runweaver_ml.dataset_management import MemoryTransport, build_datasets
from runweaver_ml.dataset_management.artifacts import ArtifactStore, BaseArtifact


class ArtifactStoreTest(unittest.TestCase):
    def test_constants_and_registration_lifecycle(self):
        store = ArtifactStore()
        for name, value in (("number", 7), ("empty", None), ("flag", False)):
            store.register(name, value)
            self.assertIs(store.get(name), value)
            self.assertIn(name, store)
        self.assertEqual(store.names(), ["empty", "flag", "number"])
        with self.assertRaises(RuntimeError):
            store.register("number", 8)
        store.register("number", 8, overwrite=True)
        self.assertEqual(store["number"], 8)
        store.deregister("number")
        store.deregister("number", silent=True)
        with self.assertRaises(KeyError):
            store.deregister("number")
        for access in (store.get, store.item):
            with self.assertRaises(KeyError):
                access("number")
        store.clear()
        self.assertEqual(len(store), 0)

    def test_parameterless_stateful_resolution_and_raw_identity(self):
        source = Mock(side_effect=[10, 11])
        store = ArtifactStore()
        store.register("counter", source)
        self.assertIs(store.item("counter"), source)
        source.assert_not_called()
        self.assertEqual([store.get("counter"), store.get("counter")], [10, 11])
        self.assertEqual(source.call_args_list, [call(), call()])

    def test_callable_model_is_accessible_without_evaluation(self):
        class Model:
            def __call__(self, sample):
                return sample * 2
        model = Model()
        store = ArtifactStore()
        store.register("model", model)
        self.assertIs(store.item("model"), model)
        self.assertEqual(store.item("model")(3), 6)
        with self.assertRaises(TypeError):
            store.get("model")

    def make_manager(self):
        return build_datasets({
            "requires": ["text"],
            "datasets": [{"name": name, "transport": MemoryTransport([
                {"__key__": "a", "text": "hello"}])}
                for name in ("alpha", "beta", "gamma")],
        })

    def test_shared_and_independent_distribution_preserves_identity_and_state(self):
        manager = self.make_manager()
        shared = random.Random(42)
        manager.distribute_artifact("rng", value=shared)
        resources = [dataset.artifact_store.item("rng") for dataset in manager.datasets]
        self.assertTrue(all(resource is shared for resource in resources))
        oracle = random.Random(42)
        self.assertEqual([resource.randint(0, 99) for resource in resources],
                         [oracle.randint(0, 99) for _ in resources])
        manager.distribute_artifact("independent", create=lambda dataset: random.Random(42))
        independent = [dataset.artifact_store.item("independent") for dataset in manager.datasets]
        self.assertEqual(len({id(resource) for resource in independent}), 3)
        before = independent[1].getstate()
        independent[0].randint(0, 99)
        self.assertEqual(independent[1].getstate(), before)

    def test_shared_callable_resolution_and_aggregation(self):
        manager = self.make_manager()
        source = Mock(side_effect=[1, 2, 3])
        manager.distribute_artifact("counter", value=source)
        resources = manager.aggregate_artifacts("counter")
        self.assertTrue(all(resource is source for resource in resources))
        source.assert_not_called()
        self.assertEqual([dataset.artifact_store.get("counter")
                          for dataset in manager.datasets], [1, 2, 3])
        self.assertEqual(source.call_args_list, [call(), call(), call()])

    def test_configuration_factory_registers_source_before_layout(self):
        class ConfiguredConstant(BaseArtifact):
            artifact_name = "phase1_test_constant"

            @classmethod
            def from_config(cls, artifact_config, dataset_config):
                return cls(name=artifact_config["name"], value=artifact_config["value"])

        manager = build_datasets({
            "requires": ["text", "language_id"],
            "artifacts": [{"name": "language_id", "handler": "phase1_test_constant", "value": 7}],
            "datasets": [{"name": "demo", "transport": MemoryTransport([
                {"__key__": "a", "text": "hello"}])}],
        })
        dataset = manager.datasets[0]
        self.assertEqual(dataset.artifact_store.item("language_id"), 7)
        self.assertEqual(list(dataset.view.iter_root())[0]["language_id"], 7)


if __name__ == "__main__":
    unittest.main()
