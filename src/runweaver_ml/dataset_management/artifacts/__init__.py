"""Dataset-level artifact interfaces and registry helpers."""

from .artifact_base import (
    BaseArtifact,
    create_artifacts,
    ARTIFACT_REGISTRY
)

from .artifact_store import ArtifactStore

__all__ = [
    "ARTIFACT_REGISTRY",
    "ArtifactStore",
    "BaseArtifact",
    "create_artifacts",
]
