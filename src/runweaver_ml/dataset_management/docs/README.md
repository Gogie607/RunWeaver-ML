# Dataset Management

Flexible multimodal WebDataset infrastructure for:
- hybrid shard discovery
- modality overlays
- dataset mixing
- progressive feature extraction
- multimodal batching
- extraction pipelines

Designed for large-scale speech and multimodal research workflows.

---

# Overview

The system provides a lightweight but extensible framework for working with multimodal WebDataset datasets.

Supported capabilities include:

- coalesced datasets
- fragmented modality shards
- hybrid overlay layouts
- dataset mixing
- weighted sampling
- per-domain iteration
- finite and looping traversal policies
- dynamic modality decoding
- progressive modality generation

The architecture is intentionally storage-oriented and stream-oriented rather than manifest/database-driven.

---

# Core Concepts

## Modalities

A modality is a named data representation such as:

- txt
- wav
- audio_encoding   (Pre encoded  wave files)
- speech_traits    (pre computed voice attributes)

Modalities are dynamically resolved and decoded at runtime.

---

## Dataset Layout Types

### Coalesced

All modalities exist in the same shard set.

Example:

```text
root/
    data/
        shard-000.tar
```

### Fragmented

Each modality exists in its own aligned shard set.

#### Example:
``` text
root/
    txt/
    wav/
    prosody/
```
#### Hybrid

Multiple modalities may coexist in partially grouped overlay shards.

The loader automatically discovers and merges aligned shards.
## Artifacts and Providers

Artifacts and providers extend a dataset with values that are not necessarily
stored as persistent modalities in its shards.

### Artifacts

Artifacts represent dataset-level values. They are created by name from the
artifact registry and may use fields from the dataset configuration. An
artifact is not sample-aware: its emitted value is obtained without passing the
current sample.

Artifacts are declared for an execution block:

```yaml
artifacts:
  - name: language
  - name: speaker_resolver
```

An artifact that emits a value may also be included in `requires`, allowing it
to appear in each runtime sample alongside persisted modalities:

```yaml
requires:
  - audio_encoding
  - speech_traits
  - language
```

Only non-sample-aware artifacts belong in `requires`. Values that depend on the
sample should be implemented as providers.

### Providers

Providers generate or resolve sample-aware runtime values. They are declared
separately from artifacts:

```yaml
providers:
  - name: speaker_id
  - name: language_id
```

Provider behavior can be configured per dataset. For example, a speaker ID can
be constant for a dataset, mapped from shard names, or extracted from the
sample key:

```yaml
datasets:
  - name: kokoro
    root: /ai/datasets/kokoro/train
    language: ja
    speaker_id:
      namespace: kokoro
      resolver:
        type: mapped
        shards:
          botchan-by-soseki-natsume-2.tar: soseki-natsume

  - name: abe
    root: /ai/datasets/abesanecdotes_2412_librivox/train
    language: en
    speaker_id:
      namespace: librivox
      resolver:
        type: constant
        speaker: abe

  - name: libritts
    root: /ai/datasets/libri_tts/train
    language: en
    speaker_id:
      namespace: libritts
      resolver:
        type: sample_key_regex
        pattern: '^(\d+)_'
        group: 1
```

### Shared Resources

Artifacts and providers may use a shared object that should be created only
once. Shared resources are registered objects constructed from configuration,
or supplied as pre-instantiated objects:

```yaml
shared_resources:
  - name: speaker_manifest
    handler: speaker_manifest
    config:
      manifest_path: /ai/datasets/speaker_ids.json
      allow_updates: false
```

In this example, speaker-related artifacts and providers can share the same
manifest rather than loading or maintaining separate copies.

The distinction is therefore:

- **Modality:** persisted sample data loaded from shards.
- **Artifact:** non-sample-aware dataset value or resolver state.
- **Provider:** sample-aware value resolved or generated at runtime.
- **Shared resource:** reusable object used by artifacts or providers.

## Dataset Manager

MultiDatasetManager controls runtime traversal semantics.

Supported mix modes:

mixed
per_domain

Supported exhaustion policies:

loop
finite_exhaustive
finite_stationary
Progressive Modality Generation

Generators can:

consume modalities
produce new modalities
inject generated data into runtime batches
optionally persist overlays as aligned shard sets

Example pipeline:
```
wav
 ├──> audio_encoding
 ├──> prosody
 └──> latent
```
Generated modalities can later be reused directly without recomputation.

Example Config
``` text
validation:

  batch_size: 16

  shuffle: false
  shardshuffle: false

  mix_mode: per_domain
  exhaust_policy: finite

  requires:
    - txt
    - audio_encoding
    - prosody

  datasets:
    - name: azurlane
      root: datasets/game_voices_azurlane/test
Runtime Batch Format
```
Batches are emitted as dictionaries:
```
{
    "sample_id": [...],
    "__shard__": [...],
    "domain": [...],

    "txt": [...],
    "audio_encoding": tensor,
    "prosody": tensor,

    "norm": tensor,
}
```
### Design Goals

The system is designed around:

streaming over indexing
shard alignment over databases
modality composition over rigid schemas
progressive enrichment workflows
low-friction experimentation

The implementation intentionally avoids:

global manifests
sample databases
heavy registries
preprocessing lock-in

#### Notes
- Internal WebDataset shard shuffling remains disabled.
- Torch DataLoader shuffling remains disabled.
- Shuffling semantics are owned by the dataset manager.
- Discovery is exploratory and tolerant of partial/corrupt experimental outputs.
- Active modality sources fail strictly once selected.
