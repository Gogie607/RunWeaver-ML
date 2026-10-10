# RunWeaver ML

**Build the training workflow you need. Keep the framework out of your research decisions.**

RunWeaver ML is a lightweight, model-agnostic Python framework for experimental machine learning. It separates two recurring engineering problems from the experiments themselves:

- **Phase Control:** coordinate multi-phase training, scheduled parameters, objectives, validation, and the training loop.
- **Dataset Management (DSM):** load samples, prepare the data your model needs, combine datasets, and deliver training batches.

**Use either subsystem independently, or use them together.** RunWeaver does not require a particular model, experiment layout, or storage format.

> **Bring your own storage backend. Keep the RunWeaver processing pipeline.**

### What does that mean in practice?

Suppose your application trains a model on audio recordings. You need to load recordings, turn them into model inputs, group them into batches, and train in several stages with different objectives. RunWeaver can handle the repeatable parts of that workflow. **You still write the model and decide how it learns.**

A few terms used throughout this project:

| Term | Plain-English meaning |
| --- | --- |
| **Transport** | How samples arrive from storage, memory, or another source. |
| **Provider** | Application-supplied processing that adds information to a sample or decides whether to keep it. For example, preparing audio features. |
| **Artifact** | A dataset-level value or resource made available to processing components. It is **not** automatically a saved model or a cached output. |
| **Collator** | Combines individual samples into a batch your training code can use. |
| **Phase** | One stage of training, with its own settings and objectives. |

You don't have to use every component. Dataset Management and Phase Control can be adopted separately.

RunWeaver began as infrastructure extracted from a research application. It remains an evolving `0.1.x` library, not a turnkey experiment runner or a frozen public API.

## Why RunWeaver?

Research code changes faster than infrastructure should. A new model might need a different objective, a training phase might freeze different components, or a dataset might move from TAR archives to a database. None of those decisions should require rewriting the entire training or input pipeline.

RunWeaver supplies the reusable machinery while **your application owns the experiment**.

| Your challenge | RunWeaver provides | You decide |
| --- | --- | --- |
| Training in distinct stages | Phase descriptions, scheduled targets, execution loops | Models, trainable components, phase assembly |
| Combining loss functions | Objective composition and shared batch context | Forward passes and loss definitions |
| Serving multimodal samples | Transport boundary, decoding, providers, prefetch, batching | Data source, modality meanings, transformations |
| Mixing datasets | Per-domain or weighted mixed iteration | Dataset selection, bias, exhaustion policy |
| Integrating into an existing project | Explicit Python construction and extension points | Configuration format, logging, checkpoints |

### Two systems, no forced marriage

```text
                         Your research application
                        /                         \
                       v                           v
              PHASE CONTROL                  DATASET MANAGEMENT
           phases · schedules             storage transport · decoding
           trainers · objectives          providers · artifacts · prefetch
           train/validate loops           mixing · collator · DataLoader
                       \                           /
                        \____ optional integration _/
```

Phase Control works with loaders supplied by your application, whether or not they came from DSM. DSM produces batches usable by your own training code, whether or not you use Phase Control.

## Install

Requires **Python 3.10+**. Declared dependencies include PyTorch, NumPy, SciPy, and WebDataset.

```bash
git clone https://github.com/Gogie607/RunWeaver-ML.git
cd RunWeaver-ML
python -m pip install -e .
```

For a non-editable local installation, use `python -m pip install .`.

## Quick start: data without TAR files

The simplest demonstration uses `MemoryTransport`, which delivers native Python values through the same DSM pipeline as a storage-backed dataset.

```python
from runweaver_ml.dataset_management import (
    MemoryTransport,
    MultimodalDataset,
    create_loader,
)

transport = MemoryTransport([
    {"__key__": "a", "text": "hello", "label": 1},
    {"__key__": "b", "text": "world", "label": 0},
])

dataset = MultimodalDataset(
    transport=transport,
    domain="demo",
    requires=["text", "label"],
)

loader = create_loader(dataset, ["text", "label"], batch_size=2)

for batch in loader:
    print(batch["sample_id"], batch["text"], batch["label"])
```

No archive files or special directory layout are needed. The transport supplies samples; RunWeaver handles the downstream pipeline.

For a custom transport, implement the public `TransportBase` contract and supply it to `MultimodalDataset` the same way. See [Transport extensions](docs/TRANSPORT_EXTENSIONS.md).

## Dataset Management: storage is only the beginning

```text
TAR / Memory / your TransportBase implementation
                     |
                     v
              LayoutSource
           decoding · identity
                     |
                     v
               DatasetView
           providers · artifacts
                     |
                     v
             threaded prefetch
                     |
                     v
          MultiDatasetManager
           per-domain or mixed
                     |
                     v
          DynamicCollator / DataLoader
```

Each `MultimodalDataset` has **one transport**. `MultiDatasetManager` can combine multiple datasets, including datasets backed by different transports.

Transports supply samples, providers can enrich or filter them, and the collator builds batches. Artifacts supply dataset-level values to components that need them. The storage format is up to the application.

Each sample needs a stable identifier (`__key__`). Other transport-specific identifiers are optional. See [Transport extensions](docs/TRANSPORT_EXTENSIONS.md) for the detailed contract.

### TAR remains a first-class option

For datasets stored in TAR archives, the included `TarTransport` can combine matching sample entries from separate modality archives.

```text
datasets/example/
├── text/
│   ├── 000.tar      # a.text.txt, b.text.txt, ...
│   └── 001.tar
└── embedding/
    ├── 000.tar      # a.embedding.npy, b.embedding.npy, ...
    └── 001.tar
```

When combining these archives, their filenames and sample identifiers must correspond.

A configuration-driven loader is available:

```python
from runweaver_ml.dataset_management import build_dataset_and_loader

config = {
    "batch_size": 8,
    "requires": ["text", "embedding"],
    "lazy": ["embedding"],
    "iteration_policy": {
        "mix_mode": "per_domain",
        "exhaust_policy": "finite",
    },
    "datasets": [
        {
            "name": "example",
            "root": "datasets/example",
            "bias": 1.0,
            "prefetch_workers": 1,
        }
    ],
}

loader = build_dataset_and_loader(config)
for batch in loader:
    print(batch["sample_id"], batch["text"])
```

In mixed mode, dataset selection weights are proportional to `sqrt(num_samples) * bias`. Exhaustion policies include finite traversal and looping options.

**Large payloads need not be loaded all at once.** TAR deferred loading indexes eligible fragments and resolves their payloads on demand. Readers have explicit shared ownership, including cleanup on failure and cancellation. Metadata counting uses raw sample identities without decoding encoding payloads, avoiding a separate eager-loading spike.

A sample batch always includes `sample_id` and `domain`; optional storage identifiers are included only when supplied. Numeric values can become tensors, arrays are stacked where appropriate, and variable-length WAV values remain lists.

## Phase Control: structure without prescribing the experiment

Phase Control describes *when and how training behavior changes*, while the host application decides *what is being trained*.

```text
PhaseManager -> PhaseContext
                     |
                     v
           targets + schedules
                     |
                     v
       ParameterWrapper / ScheduleEngine
                     |
                     v
       TrainingModule + TrainerBase
                     |
                     v
       PhaseOrchestrator.train_epoch()
         optimization · validation
         logging hooks · checkpoints
```

A phase carries a name, enabled state, run settings, targets, and schedules. The application assembles each phase's model, optimizer, loaders, and training module.

`TrainerBase` provides a small computation contract:

```python
compute(batch, params)  # -> (loss, payload)
train_context()         # context manager
eval_context()          # context manager
```

`TrainingModule` connects that contract to loop hooks and scheduled parameters. `ObjectiveComposer` is available when an experiment benefits from combining multiple `TrainingObjective` implementations around a shared `TrainingContext`; it is optional.

A configured loop can be invoked directly:

```python
from runweaver_ml.phase_control import PhaseOrchestrator, TrainLoopConfig

loop_config = TrainLoopConfig.from_config(
    {
        "schedule": {
            "log_every": 10,
            "validate_every": 100,
            "validate_max_steps": 20,
            "save_every": 500,
        },
        "amp": {
            "enabled": True,
            "dtype": "float16",
            "grad_scaler": True,
        },
    },
    max_steps=1_000,
)

final_step = PhaseOrchestrator.train_epoch(
    trainer=training_module,
    loop_config=loop_config,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=validation_loader,
    logger=logger,
)
```

The example assumes your application has constructed `training_module`, `optimizer`, and the loaders. RunWeaver does **not** construct a complete training run from YAML, choose model architectures, or impose checkpoint storage.

## Extension points

RunWeaver is intended to be extended by composition, not by copying the framework.

- **Storage:** supply a `TransportBase` implementation and a modality manifest. Native values are supported.
- **Sample processing:** install application-defined providers and artifacts.
- **Training behavior:** implement a trainer, objectives, model wrapper, and hooks as needed.
- **Configuration:** construct objects in Python or translate your own YAML/configuration into the supported dictionaries. RunWeaver itself does not load YAML.

Provider requirements are checked against the data available in the pipeline. Details of optional storage identifiers belong in the [transport documentation](docs/TRANSPORT_EXTENSIONS.md).

## Documentation

- [Architecture](ARCHITECTURE.md): subsystem boundaries, data flow, ownership, extension contracts, and limitations.
- [Transport extensions](docs/TRANSPORT_EXTENSIONS.md): the concrete public transport contract, lazy payload lifecycle, and examples.
- [TODO](TODO.md): tracked work and known discrepancies.

## Development

```bash
python -m unittest discover -s tests -v
```

The suite covers TAR behavior, transport extension cases, sample processing, resource cleanup, and metadata counting. Additional phase-control integration coverage remains a development priority.

## Scope and status

RunWeaver ML is **research infrastructure**, not an all-in-one training platform. Application-level run assembly, model and optimizer creation, concrete provider implementations, logger integration, and checkpoint destinations remain application responsibilities.

The library is currently at **0.1.x**, and its APIs may change as the framework matures. Contributions should preserve the one-way dependency: applications import RunWeaver, never the reverse.

Licensed under **GPL-3.0-or-later**. See [LICENSE](LICENSE).
