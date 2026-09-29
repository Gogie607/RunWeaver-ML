# Runweaver ML Architecture

## Scope and boundary

Runweaver ML is a standalone `src`-layout Python package containing reusable
training-control and dataset-streaming infrastructure. It may depend on the
Python standard library, its declared third-party dependencies, and other
modules within `runweaver_ml`.

The dependency direction is one-way:

```text
research application
        │
        │ constructs models, configuration, loggers, and extensions
        ▼
   runweaver_ml
```

Framework modules must not import models, paths, registries, or experiment
configuration from a host application. Application-specific behavior is
provided through objects passed into the framework or through the artifact and
provider extension registries.

## Package structure

```text
runweaver_ml
├── phase_control
│   ├── execution       trainer/module/model-wrapper contracts
│   ├── objectives      shared batch context and objective composition
│   ├── scheduling      target schedules and schedule construction
│   ├── phase_orchestrator.py
│   ├── train_phase_manager.py
│   └── train_loop_config.py
├── dataset_management
│   ├── artifacts       dataset-scoped values and artifact registry
│   ├── engine
│   │   ├── layout      iteration policy, decoding, and layout source
│   │   ├── metadata    sample-count metadata
│   │   └── transport   tar discovery and payload transport
│   ├── source          root/middleware contracts and provider registry
│   ├── views           composed sample pipeline
│   └── build_dataloader.py
└── utils               timing utilities
```

The root package exposes the package version only. Stable-looking convenience
imports live in `runweaver_ml.phase_control`, `runweaver_ml.dataset_management`,
and `runweaver_ml.utils`; lower-level transport and layout types remain internal
implementation details even though Python users can import them directly.

## Runtime overview

The training and dataset systems meet at application construction time rather
than through a hidden global runtime:

```text
tar fragments
    │
    ▼
TarTransport → LayoutSource → DatasetView → prefetch → dataset manager
                                                         │
                                                         ▼
                                                    DataLoader
                                                         │ batches
                                                         ▼
application objects → TrainingModule → PhaseOrchestrator.train_epoch
                           │                    │
                           │ parameters         ├─ optimizer / scheduler
                           │                    ├─ logging / checkpoint hook
                           ▼                    └─ validation loop
                 ParameterWrapper
                 ├─ ParamProvider
                 └─ ScheduleEngine → TargetRegistry
```

There is deliberately no package-owned dependency injection container. The
host application wires these pieces together explicitly.

## Phase control

### Phase description

`PhaseManager` validates a lightweight phase structure and produces iterable
`PhaseContext` objects. Each phase contains:

```text
name:       unique phase name
enabled:    whether the host should execute it
run:        static run fields, including steps and trainable components
targets:    scalar or boolean values eligible for scheduling
schedules:  schedule definitions keyed by target
```

`PhaseManager` stores phase contexts; it does not execute them. The host still
decides how enabled phases map to models, optimizers, datasets, and parameter
schedules.

### Runtime parameters and schedules

Runtime parameter lookup is layered:

1. `TargetRegistry` owns mutable target values.
2. `ScheduleEngine` updates registered targets for a training step.
3. `ParamProvider` resolves target values first, followed by `run`, global, and
   runtime configuration.
4. `ParameterWrapper` presents `get()`, `has()`, `step()`, and diagnostic helper
   methods to training code.

Only names already present in `TargetRegistry` can be scheduled. Registering a
schedule initializes its target at the engine's current step, which also makes
checkpoint resume state explicit.

### Trainer and objective composition

`TrainerBase` defines the central execution contract:

```text
compute(batch, params) -> (loss, payload)
train_context()        -> context manager
eval_context()         -> context manager
```

`TrainingModule` adapts that trainer to the loop lifecycle. It advances
parameters on step start, delegates computation, and provides overridable hooks
for reporting, validation metrics, collection, and checkpoints.

`ObjectiveComposer` is an optional `TrainerBase` implementation. A forward
strategy populates a per-batch `TrainingContext`, then one or more
`TrainingObjective` objects compute losses against the shared context. Their
losses are summed and their reports and validation metrics are aggregated.

The model wrapper contract controls train/eval modes and trainable component
selection without requiring the framework to understand a specific model
architecture.

### Loop ownership

`PhaseOrchestrator.train_epoch()` currently owns the inner loop:

1. enter the training context;
2. advance scheduled parameters;
3. compute loss under optional CUDA autocast;
4. backpropagate and step the optimizer (optionally with a gradient scaler);
5. step an optional learning-rate scheduler;
6. invoke payload and logging hooks;
7. periodically run validation and checkpoint hooks; and
8. stop at `max_steps`.

`validate_epoch()` runs computation without gradients in the trainer's
evaluation context and returns the module's validation summary.

`PhaseOrchestrator.run()` is currently a non-executing scaffold. Run-level
construction remains application-owned and callers must invoke the static loop
methods directly. This limitation should remain visible until a generic run
contract is designed and implemented.

## Dataset management

### Storage model

`TarTransport` discovers fragments one directory below a dataset root. A
fragment may contain one modality or several. Fragment directories participating
in a dataset must use aligned shard names, and requested fragments must contain
matching sample keys for every synchronized shard.

```text
dataset root
├── text/000.tar       s1.text.txt
├── text/001.tar       s2.text.txt
├── features/000.tar   s1.embedding.npy
└── features/001.tar   s2.embedding.npy
```

Transport produces normalized sample dictionaries. Layout decoding turns file
payloads into Python/NumPy values and adds the dataset domain. The view then
passes samples through ordered middleware providers.

Lazy loading uses `DeferredPayload` for requested single-modality fragments.
It is disabled for a fragment when multiple requested modalities in that same
fragment would require mixed eager/lazy behavior.

### Dataset pipeline

`MultimodalDataset` composes:

- a `DatasetLayout` and `TarTransport`;
- an `ArtifactStore` for dataset-scoped values;
- metadata loading or generation;
- a `DatasetView` containing the root layout source and providers; and
- a threaded `DatasetPrefetchEngine`.

Artifacts are dataset-level values or state. Providers are sample-aware
middleware that consume existing capabilities and add runtime values. Provider
installation validates both required artifacts and required input capabilities.

Artifact and provider base classes populate registries through subclass
registration. The framework contains the construction mechanism but does not
ship concrete application-specific extensions.

### Multiple datasets and batching

`MultiDatasetManager` is an `IterableDataset` with two traversal modes:

- `PER_DOMAIN`: exhaust each active dataset in order;
- `MIXED`: randomly sample active datasets using normalized
  `(num_samples ** 0.5) * bias` weights.

Exhaustion can loop, consume all datasets, or stop when the first mixed dataset
is exhausted. A domain filter can restrict active datasets.

`DynamicCollator` always emits identity fields (`sample_id`, `__shard__`, and
`domain`) and assembles requested modalities by type. It recursively collates
structured dictionaries, stacks NumPy arrays, tensorizes numeric scalars, and
leaves strings, WAV payloads, and unsupported values as lists. Missing strings,
integers, and floats receive explicit placeholder values; other partially
missing modalities remain lists.

The standard loader deliberately uses `num_workers=0`, because dataset-level
threaded prefetching is already provided by `DatasetPrefetchEngine`.

## Host application responsibilities

The application currently owns:

- loading and validating complete experiment configuration;
- constructing models and model wrappers;
- selecting trainable components;
- constructing optimizers and learning-rate schedulers;
- defining concrete objectives, artifacts, and providers;
- creating loggers and choosing checkpoint destinations;
- constructing phase-specific datasets and loaders; and
- iterating enabled phases and calling the training loop.

These concerns should move into Runweaver ML only when they can be expressed
without knowledge of a particular research project.

## Configuration boundary

Runweaver accepts Python dictionaries but does not itself load YAML or another
configuration file format. Deployment-specific configuration must not be
packaged with the library, especially absolute dataset, manifest, model, or
checkpoint paths.

Portable examples may be added as documentation or fixtures. They must use the
actual implemented schema and should be exercised by tests when practical.

## Known architectural limitations

- The public API is not yet versioned independently from the package release.
- Run-level orchestration is incomplete.
- Phase-control behavior has limited automated test coverage.
- Some contracts are abstract base classes while others remain behavioral
  conventions; protocol typing has not been standardized.
- CUDA-specific autocast and gradient-scaler calls live in the generic training
  loop, although AMP disables itself when CUDA is unavailable.
- Artifact/provider registry population depends on concrete subclasses being
  imported before configuration is built.
- Dataset metadata and synchronized tar layouts assume local filesystem access.

## Design rules

1. Preserve the one-way application-to-framework dependency.
2. Keep phase control and dataset management separately composable.
3. Prefer explicit construction and capability contracts over hidden globals.
4. Keep configuration portable and free of application paths.
5. Expose intentional convenience imports without treating every internal type
   as stable public API.
6. Fail clearly when synchronized data or required capabilities are missing.
7. Add tests before changing transport identity, schedule, or lifecycle
   semantics.
8. Do not promote the placeholder run orchestrator as a working interface.

## Verification strategy

Current tests exercise the tar transport/layout boundary, including modality
discovery, normalized decoding, multi-fragment synchronization, requested
deliverables, deferred loading, shuffle behavior, and malformed layouts.

The next highest-value coverage is:

1. parameter precedence and schedule stepping;
2. `TrainingModule` lifecycle order;
3. CPU and CUDA-disabled loop behavior;
4. multi-dataset exhaustion policies and weighting; and
5. an end-to-end example with a minimal model, dataset, optimizer, and phase.
