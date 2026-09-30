# Runweaver ML TODO

This file tracks discrepancies identified during the initial standalone-package
review. Keep an item open until its implementation, tests, and relevant
documentation have all been updated.

## Completion fields

Use these fields when closing an item:

```text
Status: Complete
Completed: YYYY-MM-DD
Commit/PR: <commit hash or pull-request URL>
Verification: <tests or manual checks performed>
Documentation updated: <README.md, ARCHITECTURE.md, API docs, or N/A>
Notes: <migration details or remaining limitations>
```

Allowed status values are `Open`, `In progress`, `Blocked`, and `Complete`.

## High priority

### TODO-001 — Correct nested parameter lookup

- Status: Complete
- Affected code: `src/runweaver_ml/phase_control/param_provider.py`
- Discrepancy: `ParamProvider.get()` checks top-level values in `run`,
  `global_cfg`, and `runtime`, but its one-level nested lookup only examines the
  last `src` left by the preceding loop. Nested values in `run` and
  `global_cfg` therefore cannot be resolved.
- Expected fix: Define and document the intended precedence, then search nested
  dictionaries in every permitted configuration layer in that order.
- Acceptance criteria:
  - Tests cover top-level and nested lookup in all three configuration layers.
  - Tests cover conflicts and confirm the documented precedence.
  - Missing keys retain the intended behavior.
- Completed: 2026-09-29
- Commit/PR: Uncommitted local change
- Verification: `test_nested_parameter_lookup_preserves_layer_precedence`
- Documentation updated: ARCHITECTURE.md
- Notes: `has()` and `get()` now share one lookup implementation and preserve
  run, global, then runtime precedence for nested values.

### TODO-002 — Align the objective metrics contract

- Status: Complete
- Affected code:
  - `src/runweaver_ml/phase_control/objectives/training_objective.py`
  - `src/runweaver_ml/phase_control/objectives/objective_composer.py`
  - `src/runweaver_ml/phase_control/execution/phase_training_module.py`
- Discrepancy: `TrainingObjective.update_metrics()` declares no payload
  parameter, while `ObjectiveComposer.update_metrics(payload)` passes one to
  every objective. `TrainingModule.on_validation_payload()` also invokes a
  payload object's `update_metrics()` method with the payload itself.
- Expected fix: Choose one explicit metric-update signature and use it
  consistently throughout the objective, composer, and module lifecycle.
- Acceptance criteria:
  - The abstract method and every caller use the same signature.
  - Validation tests exercise an objective through `TrainingModule` and
    `ObjectiveComposer` without a `TypeError`.
  - The payload ownership and metric-update flow are documented.
- Completed: 2026-09-29
- Commit/PR: Uncommitted local change
- Verification: Phase execution contract tests and full unittest suite
- Documentation updated: README.md, ARCHITECTURE.md
- Notes: Metric ownership is explicitly selected as `trainer` or `handlers`;
  objective metric updates consistently accept a payload.

### TODO-003 — Complete or remove `PhaseOrchestrator.run()`

- Status: Complete
- Affected code: `src/runweaver_ml/phase_control/phase_orchestrator.py`
- Discrepancy: `PhaseOrchestrator.run()` prints phase headings but performs no
  orchestration. Its name suggests a working high-level entry point.
- Expected fix: Either implement a generic, tested run contract or remove the
  method until that contract exists. Do not duplicate application-specific
  construction inside the framework.
- Acceptance criteria:
  - The selected behavior is covered by tests.
  - Public documentation no longer describes a placeholder as an executable
    interface.
  - Any API removal includes an appropriate migration note.
- Completed: 2026-09-29
- Commit/PR: Uncommitted local change
- Verification: Public import and full unittest suite
- Documentation updated: README.md, ARCHITECTURE.md
- Notes: Removed the nonfunctional `run()` scaffold. Applications assemble
  phases and invoke the static loop methods.

### TODO-004 — Complete the training lifecycle

- Status: Complete
- Affected code: `src/runweaver_ml/phase_control/phase_orchestrator.py`
- Discrepancy: `train_epoch()` calls `trainer.on_train_begin()` but never calls
  `trainer.on_train_end()`. Cleanup is also not protected if computation,
  logging, validation, or checkpointing raises an exception.
- Expected fix: Define whether `on_train_end()` must always run and implement
  the lifecycle with appropriate exception-safe cleanup.
- Acceptance criteria:
  - Tests assert lifecycle ordering for successful execution.
  - Tests assert the intended cleanup behavior when an exception occurs.
  - Return-step and exception semantics remain explicit.
- Completed: 2026-09-29
- Commit/PR: Uncommitted local change
- Verification: Success and exception lifecycle tests
- Documentation updated: ARCHITECTURE.md
- Notes: `on_train_end()` now runs in `finally` after training begins.

## Testing and release readiness

### TODO-005 — Add phase-control unit and integration coverage

- Status: In progress
- Affected code: `src/runweaver_ml/phase_control/`, `tests/`
- Discrepancy: Existing automated coverage is concentrated on the dataset tar
  transport/layout boundary. Parameter scheduling, lifecycle hooks, objective
  composition, and loop behavior are not covered.
- Expected fix: Add focused unit tests followed by a minimal end-to-end training
  example using a small CPU model and in-memory data.
- Acceptance criteria:
  - Tests cover parameter precedence and schedule stepping.
  - Tests cover training and validation hook order.
  - Tests cover objective loss aggregation and metric reporting.
  - Tests run without requiring CUDA.
- Completed:
- Commit/PR:
- Verification:
- Documentation updated:
- Notes:

### TODO-006 — Make device and AMP behavior explicit

- Status: Open
- Affected code:
  - `src/runweaver_ml/phase_control/phase_orchestrator.py`
  - `src/runweaver_ml/phase_control/train_loop_config.py`
- Discrepancy: The generic loop hard-codes the CUDA device type for autocast and
  gradient scaling. AMP disables itself when CUDA is unavailable, but device
  selection is not an explicit part of the loop contract.
- Expected fix: Decide whether the loop is intentionally CUDA-specific or
  accepts an explicit device/device type, then validate supported combinations.
- Acceptance criteria:
  - CPU behavior is tested.
  - Supported CUDA AMP dtypes and gradient-scaler behavior are tested or clearly
    documented.
  - Unsupported device/configuration combinations fail with actionable errors.
- Completed:
- Commit/PR:
- Verification:
- Documentation updated:
- Notes:

### TODO-007 — Formalize public API stability

- Status: Open
- Affected code:
  - `src/runweaver_ml/__init__.py`
  - `src/runweaver_ml/phase_control/__init__.py`
  - `src/runweaver_ml/dataset_management/__init__.py`
- Discrepancy: Convenience exports exist, but there is no documented stability
  policy distinguishing supported APIs from importable implementation details.
- Expected fix: Define the supported public surface and a compatibility policy
  appropriate for the `0.x` release series.
- Acceptance criteria:
  - Public exports are deliberate and import-tested.
  - README or API documentation states the compatibility expectations.
  - Deprecated names, if any, have a documented migration path.
- Completed:
- Commit/PR:
- Verification:
- Documentation updated:
- Notes:

### TODO-008 — Add a clean-environment release check

- Status: Open
- Affected code: `pyproject.toml`, CI configuration, and `tests/`
- Discrepancy: The package compiles, but the test suite was not executable in
  the review environment because declared dependencies were not installed.
  Installation and wheel contents have not yet been verified in a clean
  environment.
- Expected fix: Add CI that installs the package from `pyproject.toml`, runs the
  tests, builds a wheel, and smoke-tests imports from the built artifact.
- Acceptance criteria:
  - A clean environment can install the project and run all tests.
  - Source distribution and wheel builds succeed.
  - The wheel excludes caches, local datasets, checkpoints, and generated
    development metadata.
  - Package version metadata is consistent across the build.
- Completed:
- Commit/PR:
- Verification:
- Documentation updated:
- Notes:

## Maintenance rule

When an item is completed, retain it in this file as a short decision record.
Fill every completion field, check that referenced documentation still matches
the implementation, and create a new TODO if the fix intentionally leaves a
follow-up limitation.
