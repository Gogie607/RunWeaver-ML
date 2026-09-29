from dataclasses import dataclass, field


@dataclass
class FragmentSource:
    """
    Single discoverable fragment source.

    Examples:
        txt/
        wav/
        prosody/
        features/
    """

    id: str
    path: str
    shard_names: list[str] = field(default_factory=list)

    modalities: dict[str, str] = field(
        default_factory=dict
    )

    # optional discovery sample/cache
    sample: object | None = None


@dataclass
class TransportManifest:
    """
    Shared transport discovery state.

    Created by Transport.
    Inspected by Layout.
    Used later by Transport during iteration.
    """

    shard_names: list[str] = field(default_factory=list)

    available_modalities: dict[str, str | None] = field(
        default_factory=dict
    )

    fragments: list[FragmentSource] = field(default_factory=list)


    def get_fragment(self, fragment_id: str):
        for frag in self.fragments:
            if frag.id == fragment_id:
                return frag
        return None
