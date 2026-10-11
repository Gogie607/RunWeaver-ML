# RunWeaver ML: Architecture

RunWeaver ML extracts reusable **training control** and **dataset processing** from experimental ML applications. Its architecture deliberately avoids becoming an application framework: the host constructs models, data sources, configuration, and runtime dependencies.

The two major subsystems are **peers**, not layers of one mandatory stack.

```text
                    Research application
                  /                      \
                 v                        v
        +----------------+       +---------------------+
        | Phase Control  |       | Dataset Management  |
        | phases         |       | transports          |
        | schedules      |       | sample processing   |
        | objectives     |       | prefetch + batching |
        | training loops |       | dataset mixing      |
        +----------------+       +---------------------+
                 ^                        |
                 |       optional         |
                 +------- batches --------+
```

**Phase Control does not depend on DSM. DSM does not depend on Phase Control.** Applications can use either independently.

## Architectural principles

1. **Application-to-library dependency only.** Framework code must not import application models, paths, registries, or experiment configuration.
2. **Explicit composition.** Applications construct and connect components; there is no global runtime container.
3. **Separation of concerns.** Storage delivery, sample transformation, batch assembly, training objectives, and training execution have different owners.
4. **Extensibility at boundaries.** New transports and application-defined providers should not require modifying the processing pipeline.
5. **Preserve practical performance.** Deferred reading and bounded prefetch are part of making large datasets usable, not incidental conveniences.
6. **Make requirements local.** Optional metadata such as `__shard__` should be required only by operations that need it.

## Package map

```text
src/runweaver_ml/
├── phase_control/
│   ├── execution/          trainer, module, model wrapper contracts
│   ├── objectives/         forward strategy, objective composition, context
│   ├── scheduling/         schedule engine and schedule definitions
│   ├── train_phase_manager.py
│   ├── phase_orchestrator.py
│   ├── param_provider.py
│   ├── param_wrapper.py
│   └── train_loop_config.py
├── dataset_management/
│   ├── artifacts/          artifact values, store, registry
│   ├── source/             middleware/provider contracts and registry
│   ├── views/              DatasetView processing pipeline
│   ├── engine/
│   │   ├── transport/      public transport boundary; TAR and memory
│   │   ├── layout/         LayoutSource, decoding and iteration policy
│   │   ├── metadata/       sample counts and TAR metadata cache
│   │   ├── multimodal_dataset.py
│   │   ├── multi_dataset_manager.py
│   │   ├── dataset_prefetch_engine.py
│   │   └── dynamic_collator.py
│   └── build_dataloader.py
└── utils/
```

The root package exposes its version. Public convenience imports are available from `runweaver_ml.phase_control` and `runweaver_ml.dataset_management`. Not every class in `engine/` is a supported extension API.

---

# Part I: Phase Control

## Responsibility boundary

Phase Control coordinates training behavior over time. It does **not** choose datasets, instantiate model architectures, build optimizers, or create an application-wide experiment runner.

The host owns phase-by-phase construction. It can use RunWeaver's static loop methods with any compatible loaders, including loaders unrelated to DSM.

## Phase descriptions and execution context

`PhaseManager` validates phase entries and produces iterable `PhaseContext` objects. A phase has the following logical fields:

```yaml
name: alignment
enabled: true
run:
  steps: 1000
  trainable: [projector]
targets:
  alignment_weight: 1.0
schedules: {}
```

This illustrates the **phase structure**, not a YAML file loader. RunWeaver accepts Python dictionaries; an application can parse YAML itself.

`PhaseContext` contains phase identity, run settings, target values, schedule definitions, and pass-through global/runtime context. It does not execute the phase.

Sequential phases do not automatically share optimizers, learning-rate schedulers, or execution state. The application decides what to rebuild and what to preserve.

## Runtime parameters and scheduling

```text
PhaseContext
  ├── run settings -----------------------┐
  ├── target values -> TargetRegistry ----┤
  └── schedules -> ScheduleEngine --------┤
                                         v
                                    ParamProvider
                                         |
                                         v
                                  ParameterWrapper
                                         |
                                         v
                                training computation
```

- `TargetRegistry` stores mutable, schedule-eligible targets.
- `ScheduleEngine` updates registered targets for a given step.
- `ParamProvider` resolves target values before run/global/runtime configuration.
- `ParameterWrapper` presents `get()`, `has()`, `step()`, and diagnostics to training code.

A schedule can only update an existing registered target. Scheduling is phase-local; it is not a hidden global training state.

## Trainer and objective composition

`TrainerBase` defines the core training contract:

```python
compute(batch, params)  # -> (loss, payload)
train_context()         # -> context manager
eval_context()          # -> context manager
```

`TrainingModule` adapts the trainer to lifecycle hooks: scheduled parameter advancement, logging payloads, validation reporting, collection, and checkpoint callbacks.

For experiments with multiple losses, `ObjectiveComposer` is an **optional** `TrainerBase` implementation. A `ForwardStrategy` populates a per-batch `TrainingContext`; individual `TrainingObjective` objects compute losses from that context. The composer combines their losses and reports.

Model wrapper contracts provide mode changes and trainable-component selection without imposing a specific neural-network architecture.

## Execution loop ownership

`PhaseOrchestrator.train_epoch()` is the existing inner training loop. At a high level it:

1. Enters the training context and advances scheduled parameters.
2. Computes `(loss, payload)` under optional CUDA autocast.
3. Performs backpropagation with optional gradient accumulation and gradient scaling.
4. Updates the optimizer and optional learning-rate scheduler at accumulation boundaries.
5. Invokes payload/logging hooks.
6. Runs configured validation and checkpoint hooks.
7. Stops at the configured step limit.

`validate_epoch()` runs computation without gradients in the evaluation context and returns a validation summary.

Validation metrics may be owned by the trainer or by `TrainingModule` handlers, selected during module construction. Checkpoint persistence is injected via callbacks rather than embedded in a model wrapper.

**The host remains responsible** for creating the `TrainingModule`, optimizer, loaders, logger, and phase-specific runtime. `TrainLoopConfig` configures the loop; it is not a complete run configuration.

`TrainLoopConfig.from_config()` accepts an optional
`gradient_accumulation: {enabled: true, count: N}` runtime section. Missing
configuration disables accumulation; enabling it requires a positive integer
count. Direct construction uses `accumulation_steps=1` by default. Accumulation
works with existing float16/GradScaler and bfloat16 CUDA AMP paths.

Phase-local steps, parameter schedules, logs and validation intervals count
batches, preserving the existing contract. Losses are averaged over batches in
each group, not weighted by sample count. Optimizer and LR scheduler updates
occur once per group; a GradScaler-skipped optimizer update does not advance the
LR scheduler. Normal completion applies and correctly normalizes a final partial
group. Failure discards pending gradients. Checkpoint requests wait until an
update boundary, coalescing requests within the same group; validation retains
its batch cadence and observes the most recent completed update, potentially
before the final partial group is flushed. Accumulation state belongs to one
loop invocation and does not imply cross-phase optimizer or gradient continuity.

---

# Part II: Dataset Management (DSM)

## End-to-end data flow

```text
       +---------------------------------------+
       | Storage transport                     |
       | TarTransport / MemoryTransport / own   |
       +--------------------+------------------+
                            |
                    raw sample dictionaries
                            |
                            v
                       LayoutSource
                 identity + decoding policy
                            |
                            v
                        DatasetView
                 ordered middleware providers
                   + dataset artifact access
                            |
                            v
                    DatasetPrefetchEngine
                            |
                            v
                 MultimodalDataset iterator
                            |
                +-----------+-----------+
                |                       |
                v                       v
        standalone loader       MultiDatasetManager
                                      |
                                DynamicCollator
                                      |
                                   DataLoader
```

The diagram describes responsibilities, not separate processes. Prefetch uses threads; the standard PyTorch DataLoader uses `num_workers=0`.

## 1. Transport: where samples come from

**Transport is a storage boundary, not a processing pipeline.**

`MultimodalDataset` accepts either a local root (constructing `TarTransport` by default) or a supplied transport instance.

The public extension types are:

```python
from runweaver_ml.dataset_management import (
    TransportBase,
    TransportManifest,
    DeferredPayload,
    MemoryTransport,
)
```

A transport exposes its `manifest` before construction and implements:

```python
iter_samples(
    *,
    deliverables,
    load_policy,
    shardshuffle=False,
    anchor_modality=None,
)
```

Each call must support fresh iteration. The transport owns sample discovery, synchronization, storage access, and resource lifetime. Unsupported partition or anchor operations should fail explicitly.

The manifest's `available_modalities` maps normalized names to serializer identifiers. `None` means a native Python value; other identifiers select the existing decoding codecs.

Each emitted sample requires `__key__`, a stable identity. `__shard__` is **optional partition identity**, not necessarily a TAR filename. It is required only by explicitly shard-aware shuffle or provider operations. Transports do not have to invent partitions.

**One MMDS instance uses one transport.** DSM can combine multiple MMDS instances with different transports.

`MemoryTransport` is a reference implementation using native Python values. A contributor can implement another transport without registering it in a central backend registry.

See [Transport extensions](docs/TRANSPORT_EXTENSIONS.md) for the full public contract and compatibility details.

## 2. TAR transport: one concrete backend

`TarTransport` discovers fragments one directory below a dataset root:

```text
dataset root/
├── text/
│   ├── 000.tar
│   └── 001.tar
└── features/
    ├── 000.tar
    └── 001.tar
```

A fragment may contain one or several modalities. Requested fragments are synchronized by shard name and sample key. The transport detects missing aligned samples rather than silently inventing them.

TAR discovery, filesystem validation, fragment overlays, and payload access belong to `TarTransport`. There is **no longer a `DatasetLayout` wrapper**.

### Deferred payloads and resource ownership

Eligible heavy TAR fragments can be indexed without eagerly materializing their payload bytes. `DeferredPayload` resolves requested data later, before provider filtering.

For lazy TAR fragments:

- Indexed members refer to a shared TAR reader.
- Outstanding payloads share **reference-counted ownership** of that reader.
- Access to the reader is synchronized to protect seek and decompression state.
- Loading, failed loading, explicit abandonment, and collection release payload ownership.
- The reader closes after the final outstanding owner releases it.
- Delivered payloads may remain valid after their source iterator closes.
- Undelivered payloads are released when iteration is cancelled.
- A failed deferred load is terminal; retry requires a fresh payload from another iteration.

`DeferredPayload.close()` abandons an unresolved payload. Cached successfully loaded values remain available.

The eager WebDataset path also has explicit stream cleanup, including exception paths. Prefetch shutdown closes its active root iterator.

This lifetime design avoids per-payload archive reopening. That matters particularly for `.tar.gz`, where backward access can incur substantial decompression costs. The current synchronization path may deliver indexed samples in reverse order; compressed archive access order remains a performance consideration.

### Metadata counting

Default root-based TAR construction uses a local JSON metadata cache. When missing, metadata generation counts **raw sample identities using deferred loading**, then releases unused loaders without decoding payloads.

This avoids eagerly loading whole encoding shards merely to determine sample counts. The cache format and the historical first-discovered-modality counting universe are unchanged.

For supplied transports, MMDS accepts an explicit `num_samples`, can use the transport's `num_samples` for the default universe, or counts by raw iteration. Counting can be expensive for transports without cheap reiteration; unbounded streams are not currently an established use case.

Counting samples does not validate serialized payload contents.

## 3. LayoutSource and DatasetView: processing, not storage

`LayoutSource` consumes a transport manifest, applies decoding policy, and maps `__key__` to the public `sample_id` while adding the dataset `domain`. Optional `__shard__` is preserved only when supplied.

`DatasetView` passes samples through ordered middleware providers. Providers are application-defined and can enrich, filter, or otherwise process samples.

`build_dataset(..., providers=...)` constructs and installs configured providers
in declaration order using the existing provider registry. `build_datasets()`
delegates its group-level provider configuration to that singular builder for
each dataset, creating separate provider instances. Provider entries use
`name`, optional `handler`, and constructor settings under `config`;
application-defined provider modules must be imported before building datasets.

Provider installation checks required input capabilities and artifacts. A provider that needs partition identity declares it explicitly, for example:

```python
requires = ["text", "__shard__"]
```

The `__shard__` requirement is checked per sample before the provider processes it. Ordinary providers and transports are not forced to provide shard metadata.

Artifacts are dataset-scoped values managed by an `ArtifactStore`. Providers are sample-aware middleware. The framework supplies registries and contracts, not built-in application-specific persistent providers.

`store.register(name, source, overwrite=False)` stores the original Python object.
`store.get(name)` returns a constant or invokes a callable with no arguments on
each access; `store.item(name)` returns the original object without evaluating
it. Callable models requiring inputs should be accessed with `item()`. Missing
names raise `KeyError`; duplicate registration requires explicit overwrite.
State and resource identity belong to the stored source. Distribution preserves
shared references with `value=...`, or constructs per-dataset resources with
`create(dataset)`.

Layout `requires` selects transport and artifact fields; selected artifacts are
resolved during sample construction and take precedence over transport fields.
Unselected artifacts are not evaluated. Provider `artifact_requires` validates
presence during installation without evaluating or selecting those artifacts.
The dataset default selects transport fields; direct layout default selection
includes all available fields. `BaseArtifact` remains a configuration construction
record (`name`, `source`, `from_config`), with the registry and factory retained.
The former `Artifact` runtime wrapper has been removed. Consumers must replace
wrapper access with `get()` for values or `item()` for resources; registration
now takes a name and source. `RuntimeArtifactRepository` is unchanged.


Storage-aware application providers may access a concrete transport's own storage information, but **a local root path is not part of the generic transport contract**.

### Composition and extensibility

RunWeaver provides mechanisms for assembling and processing data without prescribing the application's interpretation of that data.

Transports supply available modalities. Artifacts supply named values or resources. Layout requirements select the fields entering the sample stream. Providers extend or transform that stream.

The framework guarantees defined interfaces and lifecycle rules: artifact value resolution invokes callables without arguments, layout requirements select sample fields, and configured providers process samples in declaration order. The application decides what an artifact represents, what a provider does, and how these components are combined.

Artifacts are not restricted to metadata, providers are not restricted to particular transformations, and sample composition is not limited to predefined recipes. Applications are free to combine these mechanisms in ways the framework's authors did not anticipate, provided they respect those contracts. Examples illustrate possible compositions, not an exhaustive list of permitted uses.

## 4. Prefetch, dataset mixing, and collation

`DatasetPrefetchEngine` uses threaded prefetch. It can decode samples ahead of consumption, improving overlap at the cost of queue memory. Prefetch workers may reorder sample delivery; strict global ordering should not be assumed.

`MultiDatasetManager` is a PyTorch `IterableDataset` supporting:

- `PER_DOMAIN`: exhaust active datasets in order.
- `MIXED`: select datasets using normalized `sqrt(num_samples) * bias` weights.

Exhaustion policies control whether traversal loops, consumes the available datasets, or stops based on the first exhausted mixed dataset. Active-domain filtering is supported.

`DynamicCollator` always includes `sample_id` and `domain`. It includes `__shard__` only when supplied, retaining aligned `None` values for missing entries in mixed batches. It handles dictionaries recursively, stacks compatible NumPy arrays, tensorizes numeric scalars, and leaves strings, WAV payloads, and unsupported objects in lists.

The standard `create_loader()` uses PyTorch DataLoader with `num_workers=0`, since DSM already provides threaded prefetch.

---

# Part III: Integration and ownership

## Who builds what?

| Concern | Owner |
| --- | --- |
| Model architectures and weights | Application |
| Optimizers and learning-rate scheduler instances | Application |
| Experiment/YAML configuration | Application |
| Phase definitions and parameter scheduling machinery | Phase Control |
| Training and validation loop implementation | Phase Control |
| Concrete objectives and forward strategy | Application extension |
| Storage access and sample universe | Transport |
| Sample decoding and pipeline composition | DSM |
| Concrete providers and artifacts | Application extension |
| Prefetch, mixing, and collation | DSM |
| Logger and checkpoint destinations | Application |
| Run-level orchestration across phases | Application |

Applications may connect the systems:

```text
application config
    ├── constructs datasets / loaders through DSM
    ├── constructs model, optimizer, trainer
    ├── constructs phase contexts and schedules
    └── invokes PhaseOrchestrator with prepared objects
```

But that integration is **a choice made by the application**, not a framework-imposed coupling.

## Configuration boundary

RunWeaver consumes Python objects and dictionaries. It does not load YAML itself and should not package machine-specific paths to datasets, models, manifests, or checkpoints.

Builders support root-based TAR datasets and explicitly supplied transport objects. Transport instances in Python dictionaries are **not** YAML backend selectors.

## Public API and compatibility

The library is at `0.1.x`; public APIs are evolving. Intentional convenience imports exist at the subsystem package level.

Notable DSM evolution:

- `DatasetLayout` was removed; `LayoutSource` remains.
- `MultimodalDataset` accepts injected transports.
- `__shard__` is optional except where explicitly required.
- TAR retains shard-aware persistence and shuffling behavior.
- TAR deferred loading has explicit shared reader ownership.
- Missing TAR metadata no longer forces eager payload extraction.

Application code depending on internal layout objects or `view.root.layout.root` needs to use the concrete transport interface appropriate to that application. Generic transports do not promise a filesystem root.

## Verification and limitations

Tests cover transport extension, TAR discovery and synchronization, lazy payloads, counting, cancellation and resource cleanup, provider interactions, and DataLoader integration. Phase Control has less automated integration coverage than DSM.

Current boundaries and limitations include:

- No built-in application-wide run assembler.
- No bundled application-specific providers or artifacts.
- No built-in SQL transport or general unbounded-stream semantics.
- Prefetch can reorder delivery and hold decoded payloads in memory.
- Compressed TAR backward seeks can be costly.
- Some contracts are behavioral rather than standardized Python protocols.
- Artifact/provider registration depends on concrete subclasses being imported.
- The generic training loop uses CUDA-specific AMP paths when enabled, with CUDA availability checks.

The guiding rule for future work is straightforward:

**Extend storage without rewriting sample processing. Extend training behavior without embedding an experiment inside the framework.**
