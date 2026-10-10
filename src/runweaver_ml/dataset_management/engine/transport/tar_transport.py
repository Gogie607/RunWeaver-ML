
from contextlib import closing
from threading import Lock
from weakref import finalize

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


class _DeferredTarReader:
    """One indexed reader, shared only while its payload loaders need it."""

    def __init__(self, archive):
        self.archive = archive
        self.lock = Lock()
        self.references = 1  # Index construction owns the initial reference.

    def acquire(self):
        with self.lock:
            self.references += 1

    def release(self):
        with self.lock:
            self.references -= 1
            if self.references == 0:
                self.archive.close()

    def read(self, member):
        # TarFile (especially its gzip cursor) is not safe for concurrent seeks.
        with self.lock:
            with self.archive.extractfile(member) as payload:
                return payload.read()


class _DeferredTarLoader:
    def __init__(self, reader, member):
        self.reader = reader
        self.member = member
        reader.acquire()
        # The callback retains the reader, never this loader. Abandoned loaders
        # release ownership even when no explicit payload.close() was requested.
        self._cleanup = finalize(self, reader.release)

    def __call__(self):
        return self.reader.read(self.member)

    def close(self):
        self._cleanup()
        self.reader = None


class TarTransport(TransportBase):

    TRANSPORT_NAME = "tar"

    SUPPORTED_EXTENSIONS = (
        ".tar",
        ".tar.gz",
    )
    def __init__(self, root):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root must be a local directory: {self.root}")
        self.manifest = self.build_manifest()

    #-----------------------------------
    def iter_samples(
            self,
            *,
            deliverables,
            load_policy,
            shardshuffle=False,
            anchor_modality=None,
    ):
        shard_names = list(self.manifest.shard_names)
        fragments = self._ordered_active_fragments(
            deliverables=deliverables,
            anchor_modality=anchor_modality,
        )

        if shardshuffle:
            random.shuffle(shard_names)

        for shard_name in shard_names:
            yield from self._iter_shard(
                shard_name=shard_name,
                deliverables=deliverables,
                load_policy=load_policy,
                fragments=fragments,
            )

    # -------------------------------------------------
    def _ordered_active_fragments(
            self,
            deliverables,
            anchor_modality=None,
    ):
        deliverables = set(deliverables or [])
        fragments = [
            fragment
            for fragment in self.manifest.fragments
            if set(fragment.modalities) & deliverables
        ]

        if anchor_modality is None:
            return fragments

        if anchor_modality not in deliverables:
            raise ValueError(
                "Tar transport anchor modality must be one of the requested "
                f"deliverables: '{anchor_modality}'."
            )

        anchor_fragments = [
            fragment
            for fragment in fragments
            if anchor_modality in fragment.modalities
        ]

        if not anchor_fragments:
            raise ValueError(
                "Tar transport anchor modality is not available in an active "
                f"fragment: '{anchor_modality}'."
            )

        if len(anchor_fragments) > 1:
            fragment_ids = [fragment.id for fragment in anchor_fragments]
            raise RuntimeError(
                "Tar transport anchor modality must identify exactly one "
                f"fragment. '{anchor_modality}' occurs in {fragment_ids}."
            )

        anchor = anchor_fragments[0]
        return [anchor] + [
            fragment for fragment in fragments
            if fragment is not anchor
        ]

    #-----------------------------------
    def _iter_source_shard(
            self,
            fragment:FragmentSource,
            shard_name
    ):

        tar_path = Path(fragment.path) / shard_name
        yield from self._iter_tar_samples(tar_path)

    @staticmethod
    def _iter_tar_samples(tar_path):
        """Keep WebDataset grouping while explicitly owning its local stream.

        WebDataset 1.0.2's StreamingOpen has no closing finally block and may
        attach the stream to exception.args. Supply our context-managed stream
        instead, so even retained WebDataset exceptions cannot keep it open.
        """
        with open(tar_path, "rb") as stream:
            def open_owned_stream(urls):
                for source in urls:
                    url = source["url"]
                    yield {"url": url, "stream": stream, "local_path": url}

            dataset = wds.WebDataset(
                [str(tar_path)], shardshuffle=False, nodesplitter=None,
                workersplitter=None, empty_check=False, cache_size=0,
            )
            for index, stage in enumerate(dataset.pipeline):
                if isinstance(stage, wds.cache.StreamingOpen):
                    dataset.pipeline[index] = open_owned_stream
                    break
            else:
                raise RuntimeError("WebDataset local stream opener was not found")

            iterator = iter(dataset)
            try:
                yield from iterator
            except Exception as exc:
                # Preserve error type/message while removing WebDataset's stream
                # argument. Tracebacks may still reference the already closed file.
                exc.args = tuple(str(tar_path) if arg is stream else arg for arg in exc.args)
                raise
            finally:
                iterator.close()
                dataset.close()

    # -------------------------------------------------
    def _iter_shard(
            self,
            shard_name:str,
            deliverables,
            load_policy,
            fragments,
    ):
        source_maps = self._read_shard_by_key(
            shard_name,
            deliverables,
            load_policy,
            fragments,
        )

        if not source_maps:
            return

        pending = []
        try:
            # choose synchronization source
            first = source_maps[0]
            while first:
                key, sample = first.popitem()
                pending.append(sample)
                merged = dict(sample)
                for other in source_maps[1:]:
                    overlay = other.pop(key, None)
                    if overlay is None:
                        raise RuntimeError(
                            f"Missing synchronized sample '{key}' in shard '{shard_name}'."
                        )
                    pending.append(overlay)
                    merged.update({
                        k: v for k, v in overlay.items() if not k.startswith("__")
                    })
                self._normalize_sample_keys(merged)
                merged["__shard__"] = shard_name
                # Ownership passes to the caller before yielding. Closing this
                # iterator must never abandon already delivered payloads.
                pending.clear()
                yield merged
        finally:
            self._close_deferred_samples(pending)
            for source_map in source_maps:
                self._close_deferred_samples(source_map.values())

    @staticmethod
    def _close_deferred_samples(samples):
        for sample in samples:
            for value in sample.values():
                if isinstance(value, DeferredPayload):
                    value.close()

    # -------------------------------------------------
    def _read_shard_by_key(
            self,
            shard_name:str,
            deliverables,
            load_policy,
            fragments=None,
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


        try:
            for fragment in (
                self.manifest.fragments
                if fragments is None
                else fragments
            ):

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
        except BaseException:
            for source_map in source_maps:
                self._close_deferred_samples(source_map.values())
            raise

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

        with closing(self._iter_source_shard(fragment, shard_name)) as samples:
            for sample in samples:
                key = sample["__key__"]
                if key in by_key:
                    raise RuntimeError(
                        f"Duplicate sample key '{key}' in shard '{shard_name}'."
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
        by_key = {}
        reader = _DeferredTarReader(tarfile.open(tar_path, "r:*"))
        try:
            for member in reader.archive.getmembers():
                if not member.isfile():
                    continue
                parsed = self._parse_member_name(member.name)
                if parsed is None:
                    continue
                key, member_modality, payload_key = parsed
                if member_modality != modality:
                    continue
                by_key.setdefault(key, {"__key__": key})
                if payload_key in by_key[key]:
                    raise RuntimeError(
                        f"Duplicate payload '{payload_key}' for key '{key}' "
                        f"in shard '{shard_name}'."
                    )
                by_key[key][payload_key] = DeferredPayload(_DeferredTarLoader(reader, member))
        except BaseException:
            self._close_deferred_samples(by_key.values())
            raise
        finally:
            reader.release()
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

        with closing(TarTransport._iter_tar_samples(first_tar)) as samples:
            return next(samples)
