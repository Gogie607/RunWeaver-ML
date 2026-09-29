from abc import abstractmethod

from typing import Iterable

from .datasource_base import MiddlewareSourceBase


PROVIDER_REGISTRY: dict[str, type["ProviderBase"]] = {}


class ProviderBase( MiddlewareSourceBase):

    """
    Base class for ordered sample-processing stages installed in a DatasetView.

    Providers:
        - receive each sample
        - may access the owning DatasetView as execution context
        - may read objects from view.artifact_store
        - may add, modify, or remove sample fields

    `requires` describes sample fields expected by the provider.
    `artifact_requires` describes named Artifacts expected in the view store.
    `capabilities` describes sample fields the provider can supply.
    """


    # ---------------------------------------------------------

    requires: list[str] = []
    artifact_requires: list[str] = []
    capabilities: list[str] = []
    provider_name: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "provider_name", None)
        if name:
            if name in PROVIDER_REGISTRY:
                raise RuntimeError(f"Provider type '{name}' is already registered.")
            PROVIDER_REGISTRY[name] = cls

    def __init__(self):
        super().__init__()

        self._capabilities = list(self.capabilities)
        self._requires = list(self.requires)

    @abstractmethod
    def process(
        self,
        sample: dict,
        view,
    ) -> dict | None:
        """
        Process one sample.

        Return:
            modified sample, or None to remove it from the stream.
        """
        raise NotImplementedError

    def iterate(
        self,
        stream: Iterable,
        ctx,
    ):
        for sample in stream:
            sample = self.process(sample, ctx)

            if sample is not None:
                yield sample

    def request(
            self,
            request,

    ) -> list[str]:
        raise RuntimeError(
            "Providers are installed explicitly and do not negotiate requests."
        )


def create_providers(provider_configs: list[dict] | None) -> list[ProviderBase]:
    providers = []
    for provider_config in provider_configs or []:
        name = provider_config["name"]
        selector = provider_config.get("handler", name)
        provider_cls = PROVIDER_REGISTRY.get(selector)
        if provider_cls is None:
            provider_cls = next(
                (
                    candidate
                    for candidate in PROVIDER_REGISTRY.values()
                    if candidate.__name__ == selector
                ),
                None,
            )
        if provider_cls is None:
            available = ", ".join(sorted(PROVIDER_REGISTRY)) or "<none>"
            raise ValueError(
                f"Unknown provider handler '{selector}' for '{name}'. "
                f"Registered providers: {available}"
            )

        providers.append(provider_cls(**provider_config.get("config", {})))

    return providers
