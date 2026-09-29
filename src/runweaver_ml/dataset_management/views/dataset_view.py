
from ..source.datasource_base import  MiddlewareSourceBase

class DatasetView:

    def __init__(
            self,
            root_source,
            runtime_artifact_repository=None,
    ):
        # iterable origin
        self.root = root_source
        self.runtime_artifact_repository = runtime_artifact_repository

        # middleware only
        self.pipeline = []

        self._requires = sorted(
            set(root_source.requires)
        )

    @property
    def deliverables(self) -> list[str]:
        out = list(self.root.deliverables)

        for provider in self.pipeline:
            out.extend(provider.capabilities)

        return list(dict.fromkeys(out))
    @property
    def capabilities(self) -> list[str]:
        out = list(self.root.capabilities)

        for provider in self.pipeline:
            out.extend(provider.capabilities)

        return list(dict.fromkeys(out))

    @property
    def requires(self) -> list[str]:
        return self._requires

    # -------------------------------------
    def add_source(
            self,
            source: MiddlewareSourceBase
    ):

        self.pipeline.append(source)


        self._requires = sorted(
            set(self._requires)
            | set(source.requires)
        )

    @property
    def artifacts(self):
        return self.root.artifact_store

    # =====================================================
    # EXECUTION
    # =====================================================

    def iter_root(self):
        """
        Iterate synchronized and decoded base samples before middleware.
        """
        return iter(self.root)


    def process_sample(self, sample):
        """
        Apply every middleware stage in installation order.
        """

        stream = iter((sample,))

        for stage in self.pipeline:
            stream = stage.iterate(
                stream,
                self
            )

        try:
            return next(stream)
        except StopIteration:
            return None


    def __iter__(self):

        for sample in self.iter_root():

            sample = self.process_sample(sample)

            if sample is not None:
                yield sample
