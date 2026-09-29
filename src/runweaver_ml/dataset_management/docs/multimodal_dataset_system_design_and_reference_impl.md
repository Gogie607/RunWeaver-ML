# Multimodal Dataset System

A lightweight aligned-modality dataset infrastructure for Ai model training and experimentation.

Designed specifically for:

- multimodal concept-model probing
- lazy modality loading
- aligned tar shard datasets
- minimal DRAM usage
- temporal feature preservation
- rapid experimentation

This is NOT production infrastructure.

---

# Design Principles

## 1. Identity First

The dataset streams identities only.

No tensors are loaded during iteration.

Dataset items:

```python
{
    "__key__": sample_id
}
```

---

## 2. Structural Alignment

Dataset integrity is encoded structurally.

Example:

```text
features/tar01.tar
latent/tar01.tar
txt/tar01.tar
```

All aligned shard names contain identical sample keys.

No embedded sample_id field required.

---

## 3. Lazy Modality Loading

Modalities are only loaded when explicitly requested.

Example:

```yaml
requires:
  - txt
  - latent
```

Only those modalities are materialized.

---

## 4. Separation of Responsibilities

## Dataset
Knows:
- sample ids
- shard mapping
- modality structure

Does NOT:
- load tensors
- decode files
- batch data

---

## Resolver
Knows:
- tar handles
- modality decoding
- extraction
- caching

---

## Collator
Knows:
- requested modalities
- batch assembly
- tensor stacking

---

# Recommended Folder Layout

```text
sandbox/
    datasets/
        dataset_layout.py
        identity_dataset.py
        modality_resolver.py
        dynamic_collator.py
        tar_cache.py
        modality_registry.py
```

---

# Dataset Config

Example:

```yaml
system:
  batch_size: 8


datasets:

  root: PROJECT/datasets/sin2pi

  modalities:
    - txt
    - latent
    - targets

  requires:
    - txt
    - latent

  validation:
    - split: val
```

---

# Dataset Layout

The layout object performs:

- shard discovery
- sample indexing
- modality validation
- tar member indexing

It owns ALL metadata.

---

# dataset_layout.py

```python
import os
import tarfile
from pathlib import Path


class DatasetLayout:

    def __init__(
        self,
        root,
        modalities,
        validate=True,
    ):

        self.root = root
        self.modalities = modalities

        self.shard_paths = {}

        self.sample_to_shard = {}

        self.tar_members = {}

        self._discover_shards()

        self._build_index()

        if validate:
            self._validate_modalities()

    # -------------------------------------------------
    # discover tar shards
    # -------------------------------------------------

    def _discover_shards(self):

        for modality in self.modalities:

            mod_root = os.path.join(self.root, modality)

            shards = sorted([
                os.path.join(mod_root, f)
                for f in os.listdir(mod_root)
                if f.endswith('.tar')
            ])

            self.shard_paths[modality] = shards

    # -------------------------------------------------
    # build sample index
    # -------------------------------------------------

    def _build_index(self):

        bootstrap = self.modalities[0]

        self.tar_members[bootstrap] = {}

        for shard_idx, shard_path in enumerate(self.shard_paths[bootstrap]):

            shard_name = Path(shard_path).stem

            self.tar_members[bootstrap][shard_idx] = {}

            with tarfile.open(shard_path, 'r') as tar:

                for member in tar.getmembers():

                    if not member.isfile():
                        continue

                    sample_id = Path(member.name).stem

                    self.sample_to_shard[sample_id] = shard_idx

                    self.tar_members[bootstrap][shard_idx][sample_id] = member.name

    # -------------------------------------------------
    # validate all modalities
    # -------------------------------------------------

    def _validate_modalities(self):

        bootstrap = self.modalities[0]

        ref_ids = set(self.sample_to_shard.keys())

        for modality in self.modalities[1:]:

            self.tar_members[modality] = {}

            found_ids = set()

            for shard_idx, shard_path in enumerate(self.shard_paths[modality]):

                self.tar_members[modality][shard_idx] = {}

                with tarfile.open(shard_path, 'r') as tar:

                    for member in tar.getmembers():

                        if not member.isfile():
                            continue

                        sample_id = Path(member.name).stem

                        found_ids.add(sample_id)

                        self.tar_members[modality][shard_idx][sample_id] = member.name

            missing = ref_ids - found_ids
            extra = found_ids - ref_ids

            if missing:
                raise RuntimeError(
                    f"Missing samples in modality={modality}: {len(missing)}"
                )

            if extra:
                raise RuntimeError(
                    f"Extra samples in modality={modality}: {len(extra)}"
                )

    # -------------------------------------------------

    def get_shard_idx(self, sample_id):
        return self.sample_to_shard[sample_id]

    def get_member_name(self, modality, shard_idx, sample_id):
        return self.tar_members[modality][shard_idx][sample_id]

    def __len__(self):
        return len(self.sample_to_shard)
```

---

# Identity Dataset

This dataset streams identities only.

---

# identity_dataset.py

```python
from torch.utils.data import Dataset


class IdentityDataset(Dataset):

    def __init__(self, layout):

        self.layout = layout

        self.sample_ids = list(layout.sample_to_shard.keys())

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):

        return {
            "__key__": self.sample_ids[idx]
        }
```

---

# Tar Cache

Open tar files are expensive.

Keep them open.

---

# tar_cache.py

```python
import tarfile


class TarCache:

    def __init__(self):

        self.cache = {}

    def get(self, path):

        if path not in self.cache:
            self.cache[path] = tarfile.open(path, 'r')

        return self.cache[path]

    def close(self):

        for tar in self.cache.values():
            tar.close()

        self.cache.clear()
```

---

# Modality Registry

Maps modality names to decoding functions.

---

# modality_registry.py

```python
import io
import numpy as np


# -------------------------------------------------
# loaders
# -------------------------------------------------


def load_txt(fileobj):

    return fileobj.read().decode('utf-8')



def load_npy(fileobj):

    return np.load(io.BytesIO(fileobj.read()))


# -------------------------------------------------

MODALITY_LOADERS = {
    "txt": load_txt,
    "latent": load_npy,
    "targets": load_npy,
    "features": load_npy,
}
```

---

# Modality Resolver

Responsible for:

- tar access
- member extraction
- decoding
- optional caching

---

# modality_resolver.py

```python
from modality_registry import MODALITY_LOADERS
from tar_cache import TarCache


class ModalityResolver:

    def __init__(
        self,
        layout,
        cache=None,
    ):

        self.layout = layout

        self.cache = cache

        self.tar_cache = TarCache()

    # -------------------------------------------------

    def load(self, sample_id, modality):

        key = (sample_id, modality)

        # ---------------------------------------------
        # optional object cache
        # ---------------------------------------------

        if self.cache is not None:

            hit = self.cache.get(key)

            if hit is not None:
                return hit

        # ---------------------------------------------

        shard_idx = self.layout.get_shard_idx(sample_id)

        shard_path = self.layout.shard_paths[modality][shard_idx]

        member_name = self.layout.get_member_name(
            modality,
            shard_idx,
            sample_id,
        )

        tar = self.tar_cache.get(shard_path)

        fileobj = tar.extractfile(member_name)

        if fileobj is None:
            raise RuntimeError(
                f"Could not extract {member_name}"
            )

        loader = MODALITY_LOADERS[modality]

        obj = loader(fileobj)

        # ---------------------------------------------

        if self.cache is not None:
            self.cache.put(key, obj)

        return obj
```

---

# Optional LRU Cache

Useful for:

- txt
- latent
- targets

Probably avoid caching giant temporal features initially.

---

# cache.py

```python
from collections import OrderedDict


class LRUCache:

    def __init__(self, capacity=256):

        self.capacity = capacity

        self.cache = OrderedDict()

    def get(self, key):

        if key not in self.cache:
            return None

        self.cache.move_to_end(key)

        return self.cache[key]

    def put(self, key, value):

        self.cache[key] = value

        self.cache.move_to_end(key)

        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

---

# Dynamic Collator

This is where runtime modality composition occurs.

---

# dynamic_collator.py

```python
import torch
import numpy as np


class DynamicCollator:

    def __init__(
        self,
        resolver,
        requires,
    ):

        self.resolver = resolver

        self.requires = requires

    # -------------------------------------------------

    def __call__(self, batch):

        sample_ids = [x["__key__"] for x in batch]

        out = {
            "sample_ids": sample_ids
        }

        for modality in self.requires:

            vals = [
                self.resolver.load(sid, modality)
                for sid in sample_ids
            ]

            out[modality] = self._assemble(vals)

        return out

    # -------------------------------------------------

    def _assemble(self, vals):

        v0 = vals[0]

        # numpy tensors
        if isinstance(v0, np.ndarray):

            return torch.tensor(np.stack(vals))

        # strings
        if isinstance(v0, str):
            return vals

        return vals
```

---

# Dataloader Assembly

---

# build_dataloader.py

```python
from torch.utils.data import DataLoader

from dataset_layout import DatasetLayout
from identity_dataset import IdentityDataset
from modality_resolver import ModalityResolver
from dynamic_collator import DynamicCollator



def build_probe_dataloader(cfg):

    layout = DatasetLayout(
        root=cfg["root"],
        modalities=cfg["modalities"],
    )

    dataset = IdentityDataset(layout)

    resolver = ModalityResolver(layout)

    collator = DynamicCollator(
        resolver,
        requires=cfg["requires"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )

    return loader
```

---

# Example Usage

```python
loader = build_probe_dataloader(cfg)

for batch in loader:

    txt = batch["txt"]

    latent = batch["latent"]
```

---

# Why This Architecture Works

## Minimal DRAM Usage

The dataset itself holds:

- sample ids
- shard indices
- tar member names

NOT tensors.

---

## No Duplicate Temporal Features

Features are loaded:

- only when requested
- only during batch assembly
- only once per sample access

---

## Extremely Flexible

New modality:

```text
emotion/
semantic/
mel/
```

Requires only:

```python
MODALITY_LOADERS["emotion"] = ...
```

No dataset rewrite.

---

## Compatible With Existing Layout

No reorganization required.

Current aligned shard structures remain usable.

---

## Future Extensions

Possible future additions:

- mmap npy loading
- shard-local batching
- async prefetching
- worker-local tar caches
- compressed latent formats
- temporal streaming

WITHOUT redesigning the architecture.

---

# Recommended First Milestone

Implement ONLY:

- DatasetLayout
- IdentityDataset
- ModalityResolver
- DynamicCollator

Run:

```yaml
requires:
  - txt
  - latent
```

Verify:

- stable memory
- correct modality alignment
- no duplicated tensors
- acceptable shard extraction speed

Only optimize AFTER instrumentation.

