

class RuntimeArtifactRepository:
    """
    Process-local repository for shared runtime artifacts.

    An artifact may be registered as either:

        1. A lazy artifact:
           - handler: callable used to create the artifact
           - config: configuration passed to the handler

        2. An existing instance:
           - returned directly without invoking a handler

    Lazily created artifacts are cached after their first access, giving each
    registered artifact singleton-like lifetime within this repository.
    """

    MISSING = object()

    def __init__(self, source_registry = None):
        self._artifacts = {}
        self._registry = source_registry

    def register(
        self,
        name: str,
        *,
        handler:str | object | None = None,
        config=None,
        instance=MISSING,
    ):
        if name in self._artifacts:
            raise KeyError(f"Artifact already registered: {name}")

        # if I have an instance I do not care about handler
        if instance is self.MISSING:
            if handler is None:
                raise ValueError(
                    f"Artifact '{name}' requires either a handler or an instance"
                )
            # this should be a handler class
            if isinstance(handler, str):
                # resolve
                try:
                    handler = self._registry[handler]
                except KeyError:
                    raise KeyError(
                        f"Unresolvable artifact handler: {handler}"
                    ) from None

            # if it is an object it is expected to be callable with config entries

        self._artifacts[name] = {
            "handler": handler,
            "config": config,
            "instance": instance,
        }

    def register_instance(self, name: str, instance):
        """
        Register an already-created object or a simple runtime value.

        Examples:
            repository.register_instance("seed", 1234)
            repository.register_instance("text_model", model)
        """
        self.register(name, instance=instance)

    def register_factory(self, name: str, handler, config=None):
        """
        Register an artifact that will be created on first access.
        """
        self.register(
            name,
            handler=handler,
            config=config,
        )

    def get(self, name: str):
        try:
            record = self._artifacts[name]
        except KeyError:
            raise KeyError(f"Unknown runtime artifact: {name}") from None

        if record["instance"] is not self.MISSING:
            return record["instance"]

        instance = record["handler"](record["config"])
        record["instance"] = instance

        return instance

    def contains(self, name: str) -> bool:
        return name in self._artifacts

    def is_instantiated(self, name: str) -> bool:
        try:
            record = self._artifacts[name]
        except KeyError:
            raise KeyError(f"Unknown runtime artifact: {name}") from None

        return record["instance"] is not self.MISSING