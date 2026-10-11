"""Dataset-scoped named value sources and explicit raw-object access."""


class ArtifactStore:
    """Hold original objects; resolve callable sources only on get()."""

    def __init__(self):
        self._artifacts = {}

    def register(self, name, source, overwrite=False):
        if not overwrite and name in self._artifacts:
            raise RuntimeError(f"Artifact '{name}' already registered.")
        self._artifacts[name] = source

    def get(self, name):
        """Resolve a constant or invoke a callable without arguments, each time."""
        source = self.item(name)
        return source() if callable(source) else source

    def item(self, name):
        """Return the original object without invoking it; missing names raise KeyError."""
        return self._artifacts[name]

    def has(self, name):
        return name in self._artifacts

    def names(self):
        return sorted(self._artifacts)

    def deregister(self, name, silent=False):
        if name not in self._artifacts:
            if silent:
                return
            raise KeyError(name)
        del self._artifacts[name]

    def clear(self):
        self._artifacts.clear()

    def __contains__(self, name):
        return self.has(name)

    def __getitem__(self, name):
        return self.get(name)

    def __len__(self):
        return len(self._artifacts)

    def __repr__(self):
        return f"{self.__class__.__name__}(artifacts={self.names()})"
