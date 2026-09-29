"""Dataset construction, multimodal loading, and batching utilities."""

__all__ = [
    "DatasetPrefetchEngine",
    "DatasetView",
    "DynamicCollator",
    "build_dataset",
    "build_dataset_and_loader",
    "build_datasets",
    "create_loader",
    "MultimodalDataset",
    "MultiDatasetManager",
]


def __getattr__(name):
    """Load public dataset helpers lazily and avoid package import cycles."""
    if name == "build_dataset":
        from .build_dataloader import build_dataset
        return build_dataset
    if name == "build_dataset_and_loader":
        from .build_dataloader import build_dataset_and_loader
        return build_dataset_and_loader
    if name == "build_datasets":
        from .build_dataloader import build_datasets
        return build_datasets
    if name == "create_loader":
        from .build_dataloader import create_loader
        return create_loader
    if name == "DatasetPrefetchEngine":
        from .engine.dataset_prefetch_engine import DatasetPrefetchEngine
        return DatasetPrefetchEngine
    if name == "DatasetView":
        from .views.dataset_view import DatasetView
        return DatasetView
    if name == "DynamicCollator":
        from .engine.dynamic_collator import DynamicCollator
        return DynamicCollator
    if name == "MultimodalDataset":
        from .engine.multimodal_dataset import MultimodalDataset
        return MultimodalDataset
    if name == "MultiDatasetManager":
        from .engine.multi_dataset_manager import MultiDatasetManager
        return MultiDatasetManager
    raise AttributeError(name)
