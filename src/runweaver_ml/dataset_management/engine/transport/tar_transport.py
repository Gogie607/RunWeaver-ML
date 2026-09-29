
import tarfile
import random
import  webdataset as wds

from pathlib import Path

from .transport_base import (
    TransportBase, DeferredPayload
)
from .transport_manifest import (
    TransportManifest, FragmentSource
)


class TarTransport(TransportBase):

    TRANSPORT_NAME = "tar"

    SUPPORTED_EXTENSIONS = (
        ".tar",
        ".tar.gz",
    )
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = self.build_manifest()

    #-----------------------------------
    def iter_samples(
            self,
            *,
            deliverables,
            load_policy,
            shardshuffle=False,
    ):
        shard_names = list(self.manifest.shard_names)

        if shardshuffle:
            random.shuffle(shard_names)

        for shard_name in shard_names:
            yield from self._iter_shard(
                shard_name=shard_name,
                deliverables=deliverables,
                load_policy=load_policy,
            )

    #-----------------------------------
    def _iter_source_shard(
            self,
            fragment:FragmentSource,
            shard_name
    ):

        tar_path = Path(fragment.path) / shard_name
        for sample in wds.WebDataset(
             [str(tar_path)],
             shardshuffle=False,
             nodesplitter=None,
             workersplitter=None,
             empty_check=False
        ):
            #

            yield sample

    # -------------------------------------------------
    def _iter_shard(
            self,
            shard_name:str,
            deliverables,
            load_policy
    ):
        source_maps = self._read_shard_by_key(
            shard_name,
            deliverables,
            load_policy,
        )

        if not source_maps:
            return

        # choose synchronization source
        first = source_maps[0]

        # synchronized merge
        while first:
            key, sample = first.popitem()

            merged = dict(sample)

            for other in source_maps[1:]:

                overlay = other.pop(key, None)

                if overlay is None:
                    raise RuntimeError(
                        f"Missing synchronized sample "
                        f"'{key}' in shard '{shard_name}'."
                    )

                merged.update({
                    k: v
                    for k, v in overlay.items()
                    if not k.startswith("__")
                })

            self._normalize_sample_keys(merged)
            merged["__shard__"] = shard_name

            yield merged

    # -------------------------------------------------
    def _read_shard_by_key(
            self,
            shard_name:str,
            deliverables,
            load_policy
    ):
        source_maps = []

        deliverables = set(deliverables or [])
        lazy_modalities = set(load_policy or [])

        # warn only; bad lazy names should not kill training
        known_modalities = {
            m
            for fragment in self.manifest.fragments
            for m in fragment.modalities
        }

        unknown_lazy = lazy_modalities - known_modalities
        if unknown_lazy:
            print(
                f"[WARN] Unknown lazy modalities ignored: "
                f"{sorted(unknown_lazy)}"
            )


        for fragment in self.manifest.fragments:

            fragment_deliverables = (
                set(fragment.modalities.keys())
                & deliverables
            )

            if not fragment_deliverables:
                continue

            lazy_hits = (
                    fragment_deliverables
                    & lazy_modalities
            )

            use_lazy = bool(lazy_hits)

            # Hybrid delivered fragment:
            # disable lazy because WDS/tar sample grouping would mix eager/lazy
            # behavior inside the same fragment.
            if use_lazy and len(fragment_deliverables) > 1:
                print(
                    f"[WARN] Lazy loading disabled for fragment "
                    f"'{fragment.id}' in shard '{shard_name}' because "
                    f"it contains multiple requested modalities: "
                    f"{sorted(fragment_deliverables)}"
                )
                use_lazy = False

            if use_lazy:
                by_key = self._lazy_fragment_map(
                    fragment=fragment,
                    shard_name=shard_name,
                    modality=next(iter(lazy_hits)),
                )
            else:
                by_key = self._eager_fragment_map(
                    fragment=fragment,
                    shard_name=shard_name,
                )

            source_maps.append(by_key)

        return source_maps

    # -------------------------------------------------
    @staticmethod
    def _normalize_sample_keys(sample):

        rename = []

        for key in sample.keys():

            if key.startswith("__"):
                continue

            modality, _ = TarTransport._modality_suffix(key)

            if modality != key:
                rename.append((key, modality))

        for old, new in rename:
            sample[new] = sample.pop(old)

    # -------------------------------------------------
    def _eager_fragment_map(
            self,
            fragment: FragmentSource,
            shard_name: str
    ):
        by_key = {}

        for sample in self._iter_source_shard(
                fragment,
                shard_name
        ):
            key = sample["__key__"]

            if key in by_key:
                raise RuntimeError(
                    f"Duplicate sample key '{key}' "
                    f"in shard '{shard_name}'."
                )

            by_key[key] = sample

        return by_key

    # -------------------------------------------------
    def _lazy_fragment_map(
            self,
            fragment: FragmentSource,
            shard_name: str,
            modality: str,
    ):
        """
        Build a WDS-like source map without reading payload bytes.

        This is intended for single-modality / single-deliverable
        heavy fragments such as audio_encoding.
        """

        tar_path = Path(fragment.path) / shard_name
        tar = tarfile.open(tar_path, "r:*")

        by_key = {}

        for member in tar.getmembers():

            if not member.isfile():
                continue

            parsed = self._parse_member_name(member.name)
            if parsed is None:
                continue

            key, member_modality, payload_key = parsed

            if member_modality != modality:
                continue

            if key not in by_key:
                by_key[key] = {
                    "__key__": key,
                }

            if payload_key in by_key[key]:
                raise RuntimeError(
                    f"Duplicate payload '{payload_key}' "
                    f"for key '{key}' in shard '{shard_name}'."
                )

            by_key[key][payload_key] = DeferredPayload(
                lambda tar=tar, member=member: (
                    tar.extractfile(member).read()
                )
            )

        return by_key

    # -------------------------------------------------
    @staticmethod
    def _parse_member_name(member_name: str):
        """
        Example:
            abc123.audio_encoding.npy.gz

        Returns:
            key          -> abc123
            modality     -> audio_encoding
        payload_key  -> audio_encoding.npy.gz

        payload_key intentionally mimics WDS sample dict keys so
        TarTransport._normalize_sample_keys() can still rename it.
        """

        name = Path(member_name).name
        parts = name.split(".")

        if len(parts) < 2:
            return None

        key = parts[0]
        modality = parts[1]
        payload_key = ".".join(parts[1:])

        return key, modality, payload_key

    # -------------------------------------------------
    @staticmethod
    def _modality_suffix(name):
        parts = name.split(".")
        return parts[0], parts[-1]

    # -------------------------------------------------

    def build_manifest(self) -> TransportManifest:

        manifest = TransportManifest()

        for child in sorted(self.root.iterdir()):

            if not child.is_dir():
                continue

            files = sorted(
                p for p in child.iterdir()
                if p.is_file()
                and any(
                    p.name.endswith(ext)
                    for ext in self.SUPPORTED_EXTENSIONS
                )
            )

            if not files:
                continue

            fragment = FragmentSource(
                id=child.name,
                path=str(child),
                shard_names=[p.name for p in files],
            )
            sample = self._fetch_shard_sample(fragment)
            fragment.sample = sample
            manifest.fragments.append(fragment)

        if not manifest.fragments:
            return manifest

        manifest.shard_names = list(
            manifest.fragments[0].shard_names
        )

        for fragment in manifest.fragments[1:]:
            if fragment.shard_names != manifest.shard_names:
                raise RuntimeError(
                    "Hybrid modality shard names must match within one "
                    f"dataset. Source '{fragment.path}' differs "
                    f"from '{manifest.fragments[0].path}'."
                )

        for fragment in manifest.fragments:

            modalities = {}

            for key in fragment.sample.keys():

                if key.startswith("__"):
                    continue

                modal_key, serialization = self._modality_suffix(key)

                modalities[modal_key] = serialization
                manifest.available_modalities[modal_key] = serialization

            fragment.modalities = modalities

        return manifest

    @staticmethod
    def _fetch_shard_sample(fragment:FragmentSource):
        first_tar = (
                Path(fragment.path)
                / fragment.shard_names[0]
        )

        sample = next(
            iter(
                wds.WebDataset(
                    [str(first_tar)],
                    shardshuffle=False,
                    nodesplitter=None,
                    workersplitter=None,
                    empty_check=False
                )
            )
        )
        return sample
