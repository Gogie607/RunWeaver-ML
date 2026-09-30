# Runweaver ML

Runweaver ML is a small, model-agnostic Python framework for experimental
machine-learning workflows. It provides two cooperating subsystems:

- phase-oriented training and validation utilities; and
- streaming, multimodal datasets backed by aligned tar shards.

The package was extracted from a larger research application. It is usable as a
standalone library, but its public API is still evolving and it does not yet
provide a complete application-level training runner.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- PyTorch
- WebDataset

Runweaver ML is licensed under the GNU General Public License v3.0 or later.
See [LICENSE](LICENSE) for the license notice and terms.

## Installation

Create and activate a virtual environment, then install the project in editable
mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For a regular, non-editable installation, use `python -m pip install .`.

## Package overview

```text
src/runweaver_ml/
├── phase_control/       phase definitions, schedules, objectives, and loops
├── dataset_management/ tar discovery, decoding, views, mixing, and batching
└── utils/               timing helpers
```

Frequently used imports are exposed by their owning subpackage:

```python
from runweaver_ml.phase_control import (
    PhaseOrchestrator,
    TrainLoopConfig,
    TrainingModule,
)
from runweaver_ml.dataset_management import (
    MultimodalDataset,
    build_dataset_and_loader,
)
```

The top-level `runweaver_ml` package currently exposes only `__version__`.

## Dataset quick start

Runweaver discovers tar shards beneath modality directories. Every fragment
must use matching shard names and sample keys when samples are assembled across
modalities:

```text
datasets/example/
├── text/
│   ├── 000.tar    # s1.text.txt, s2.text.txt, ...
│   └── 001.tar
└── embedding/
    ├── 000.tar    # s1.embedding.npy, s2.embedding.npy, ...
    └── 001.tar
```

The member name format is `<sample-id>.<modality>.<extension>`. Supported
decoders currently include text, NumPy (`.npy` and `.npz`), and WAV payloads.

A loader can be built from a plain dictionary:

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
    print(batch["sample_id"], batch["text"], batch["embedding"].shape)
```

`mix_mode` accepts `per_domain` or `mixed`. `exhaust_policy` accepts `finite`
(an alias for `finite_exhaustive`), `finite_exhaustive`, `finite_stationary`, or
`loop`. In mixed mode, dataset probabilities are proportional to
`sqrt(dataset_size) * bias`.

Dataset batches always contain `sample_id`, `__shard__`, and `domain`. NumPy
arrays are stacked as float32 tensors, integer and floating-point scalars become
tensors, strings remain lists, and WAV values remain lists so variable-length
audio is not stacked automatically.

## Training integration

The implemented training entry points are `PhaseOrchestrator.train_epoch()` and
`PhaseOrchestrator.validate_epoch()`. Applications construct the model,
optimizer, loaders, parameter wrapper, trainer, and optional logger before
calling these loops:

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

Here `training_module` is a `TrainingModule` wrapping a trainer whose
`compute(batch, params)` method returns `(loss, payload)` and whose model state
is managed through `train_context()` and `eval_context()`. `TrainingModule`
advances scheduled parameters at the beginning of every step and supplies
logging, validation, and checkpoint hooks.

Applications assemble each phase and call the static epoch methods directly.
Steps and parameter schedules are local to each call. The current loop is
PyTorch-oriented and uses CUDA AMP only when CUDA is available. Checkpointing
is optional and supplied as a callback when constructing `TrainingModule`;
RunWeaver does not prescribe model or training-session persistence.

Validation metric ownership is explicit. A module may use its metric and
collector handlers, or set `validation_owner="trainer"` when its trainer
implements `reset_metrics()`, `update_metrics(payload)`, and
`report_metrics()`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for component responsibilities and data
flow. Review [TODO.md](TODO.md) for known discrepancies, acceptance criteria,
and completion tracking.

## Development

Run the test suite from the repository root after installing the project and
its dependencies:

```bash
python -m unittest discover -s tests -v
```

The present tests focus on the tar transport/layout boundary: modality
discovery, synchronized fragment merging, decoding, lazy payloads, shuffling,
and failure cases.

## Project status

This is an early research framework (`0.1.x`), not a stable public API. Areas
that remain intentionally application-owned or incomplete include:

- constructing a full run from configuration;
- application-level phase assembly;
- model, optimizer, logger, and checkpoint construction;
- built-in artifact/provider implementations (the framework supplies their
  extension contracts and registries); and
- broader phase-control and integration test coverage.

Contributions should preserve the dependency boundary: applications may import
Runweaver ML, but Runweaver ML must not import application-specific code.
