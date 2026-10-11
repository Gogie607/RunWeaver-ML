
from typing import Any

# =========================================================
# Artifact
#  Simple storage class wrapper
#  around an object that contains or generates data
# required wrapper class for objects stored in ArtifactStore
#
# =========================================================

class Artifact:

    """
    Runtime state container.

    May:
        - hold static values
        - lazily compute values
        - mutate internal state
        - operationalize itself into samples

    Default behavior:
        insert artifact value into sample
        under artifact name.
    """

    # optional specialized provider
    #provider_cls = None

    # -----------------------------------------------------

    def __init__(
        self,
        name: str,
        value,
    ):

        self._name = name
        self._value = value

    # -----------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    # -----------------------------------------------------


    def value(self, *args, **kwargs) -> Any:
        # handle both value generators and static values
        if callable(self._value):
            return self._value(*args, **kwargs)

        return self._value

    # returns actual content regardless of type
    def item(self):
        return self._value
