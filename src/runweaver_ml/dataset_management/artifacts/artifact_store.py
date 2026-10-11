from .artifact import Artifact

class ArtifactStore:

    """
    Lightweight runtime artifact registry.

    Responsibilities:
        - uniqueness enforcement
        - runtime lookup
        - optional deregistration

    Artifacts are treated automically.

    Artifact names are effectively:
        runtime capabilities.
    """

    # ---------------------------------------------------------

    def __init__(self):

        self._artifacts = {}

    # ---------------------------------------------------------

    def register(
        self,
        artifact: Artifact,
        overwrite:bool=False,
    ):

        name = artifact.name

        if (
            not overwrite
            and name in self._artifacts
        ):

            raise RuntimeError(
                f"Artifact '{name}' already registered."
            )

        self._artifacts[name] = artifact

    # ---------------------------------------------------------

    def get(
        self,
        name,
        default=None,
    )->Artifact | None:

        return self._artifacts.get(
            name,
            default,
        )

    # ---------------------------------------------------------

    def has(
        self,
        name,
    ):

        return (
            name in self._artifacts
        )

    # ---------------------------------------------------------

    def names(self):

        return sorted(
            self._artifacts.keys()
        )

    # ---------------------------------------------------------

    def deregister(
        self,
        name,
        silent=False,
    ):

        if name not in self._artifacts:

            if silent:
                return

            raise RuntimeError(
                f"Artifact '{name}' not registered."
            )

        del self._artifacts[name]

    # ---------------------------------------------------------

    def clear(self):

        """
        Dangerous runtime reset.
        """

        self._artifacts.clear()

    # ---------------------------------------------------------

    def __contains__(
        self,
        name,
    ):

        return self.has(name)

    # ---------------------------------------------------------

    def __getitem__(
        self,
        name,
    ):

        artifact = self.get(name)

        if artifact is None:

            raise KeyError(name)

        return artifact

    # ---------------------------------------------------------

    def __len__(self):

        return len(
            self._artifacts
        )

    # ---------------------------------------------------------

    def __repr__(self):

        return (
            f"{self.__class__.__name__}"
            f"(artifacts={self.names()})"
        )