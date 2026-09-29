from pathlib import Path
import torch

from .layout.dataset_layout import (
    DatasetLayout,
)
from .layout.dataset_layout_source import (
    LayoutSource, LayoutOpParams)
from .transport.tar_transport import TarTransport

from .dataset_prefetch_engine import DatasetPrefetchEngine

from ..views.dataset_view import DatasetView
from ..source import MiddlewareSourceBase

from runweaver_ml.dataset_management.artifacts import (
    Artifact, ArtifactStore
)

from .metadata import (
    MetadataLoader
)

_ITERATION_END = object()


class MultimodalDataset(torch.utils.data.IterableDataset):

    def __init__(
            self,
            root,
            *,
            requires=None,
            load_policy=[],
            op_params=None,
            artifacts = None,
            runtime_artifact_repository=None,
            domain=None,
            bias: float = 1.0,
            prefetch_workers=1
    ):
        self.root = Path(root)
        self.domain = domain or self.root.name
        self.bias = bias
        self.op_params = op_params or {}
        self.prefetch_workers = prefetch_workers
        self.runtime_artifact_repository = runtime_artifact_repository
        self._wait_report = {}

        self.layout = DatasetLayout(
            domain=self.domain,
            root=self.root,
            transport=TarTransport(self.root),
        )

        # Register static artifacts before the immutable source contract is built.
        self.artifact_store = ArtifactStore()

        artifacts = list(artifacts or [])
        for artifact in artifacts:
            self.artifact_store.register(artifact)

        self.metadata = MetadataLoader.load_or_generate(self)

        print(self.layout.summary(),
              f"Total samples: {self.metadata.num_samples}")

        ls_params = LayoutOpParams(
            **self.op_params.get("layout", {})
        )

        self.view = self.create_view(
            op_params=ls_params,
            load_policy=load_policy,
            # required=list(self.layout.available_modalities.keys())
            required=(
                requires
                if requires is not None
                else list(self.layout.available_modalities.keys())
            )
        )

        self.prefetch_engine = DatasetPrefetchEngine(
            pipeline=self.view,
            queue_size=16,
            name=f"{self.domain}-prefetch",
            num_workers=prefetch_workers
        )

    # -------------------------------------------------

    def create_view(
            self, *,
            op_params: LayoutOpParams | None = None,
            load_policy=[],
            required=None,

    ) -> DatasetView:
        view = DatasetView(
            LayoutSource(
                layout=self.layout,
                load_policy=load_policy,
                deliverables=required,
                artifact_store=self.artifact_store,
                op_params=op_params or LayoutOpParams()
            ),
            runtime_artifact_repository=self.runtime_artifact_repository,
        )
        # add providers for artifacts that require it

        return view

    def install_provider(self, provider:  MiddlewareSourceBase):
        """Manually append a provider to the dataset's ordered pipeline."""

        if not isinstance(provider, MiddlewareSourceBase):
            raise TypeError("provider must be an artifact name, Artifact, or middleware")

        # check that any required artifacts are pre-installed
        missing = [
            name
            for name in provider.artifact_requires
            if not self.artifact_store.has(name)
        ]

        if missing:
            raise RuntimeError(
                f"Cannot install {type(provider).__name__} "
                f"for dataset '{self.domain}': "
                f"missing artifacts {missing}"
            )

        missing_inputs = [
            name
            for name in provider.requires
            if name not in self.view.capabilities
        ]
        if missing_inputs:
            raise RuntimeError(
                f"Cannot install {type(provider).__name__} "
                f"for dataset '{self.domain}': "
                f"missing input capabilities {missing_inputs}"
            )
        self.view.add_source(provider)
        return provider

    # new multi threaded iteration
    def __iter__(self):
        return iter(self.prefetch_engine)

    @property
    def num_samples(self):
        return self.metadata.num_samples

    # ===================
    # Artifact helper functions
    # ================
    def get_artifact_names(self, ) -> list[str]:
        return self.artifact_store.names()

    def get_artifact(
            self,
            name: str
    ) -> Artifact | None:
        return self.artifact_store.get(name)

    def get_artifact_value(
            self,
            name: str
    ) -> Artifact | None:
        a = self.artifact_store.get(name)
        return a.value() if a else None

    def get_artifact_item(
            self,
            name: str
    ) -> Artifact | None:
        a = self.artifact_store.get(name)
        return a.item() if a else None

    def register_artifacts(
            self,
            artifact: Artifact,
    ) -> None:
        self.artifact_store.register(artifact)
