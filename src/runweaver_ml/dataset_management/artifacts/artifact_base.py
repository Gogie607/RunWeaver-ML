from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .artifact import Artifact


ARTIFACT_REGISTRY: dict[str, type["BaseArtifact"]] = {}


class BaseArtifact(Artifact, ABC):
    """Configuration-created artifact with automatic type registration."""

    artifact_name: str | None = None
    requires: list[str] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "artifact_name", None)
        if name:
            if name in ARTIFACT_REGISTRY:
                raise RuntimeError(f"Artifact type '{name}' is already registered.")
            ARTIFACT_REGISTRY[name] = cls

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        artifact_config: dict[str, Any],
        dataset_config: dict[str, Any],
    ) -> "BaseArtifact":
        raise NotImplementedError


def create_artifacts(
    artifact_configs: list[dict] | None,
    dataset_config: dict,
) -> list[BaseArtifact]:
    artifacts = []
    for config in artifact_configs or []:
        name = config["name"]
        selector = config.get("handler", name)
        artifact_cls = ARTIFACT_REGISTRY.get(selector)
        if artifact_cls is None:
            artifact_cls = next(
                (
                    candidate
                    for candidate in ARTIFACT_REGISTRY.values()
                    if candidate.__name__ == selector
                ),
                None,
            )
        if artifact_cls is None:
            available = ", ".join(sorted(ARTIFACT_REGISTRY)) or "<none>"
            raise ValueError(
                f"Unknown artifact handler '{selector}' for '{name}'. "
                f"Registered artifacts: {available}"
            )
        artifacts.append(artifact_cls.from_config(config, dataset_config))

    return artifacts
