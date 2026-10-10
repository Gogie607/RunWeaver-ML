# RunWeaver Architectural Decisions

This record explains why RunWeaver works as it does. [ARCHITECTURE.md](../ARCHITECTURE.md)
describes how its components work. Design intentions below are developer-confirmed;
implementation observations are identified separately. Historical behavior may be
reconsidered deliberately, but alternatives alone do not justify changing it.

## Established decisions — developer-confirmed

- **PhaseManager is a scripting convenience.** It organizes sequential training
  configurations and exposes dictionaries through phase contexts. YAML parsing
  belongs to a separate utility. The manager neither assigns configuration
  ownership nor enforces application policy or phase continuity.
- **Configuration is intentionally flexible.** Dictionaries support experimental
  variation. Consumers interpret and validate required values; RunWeaver may
  validate internal invariants without imposing rigid application schemas.
  Accepting configuration does not guarantee meaningful or executable behavior.
- **Phases are independent.** Almost every experimental choice may change.
  Phase-specific execution infrastructure must be reconstructed for each phase;
  sequential execution does not imply reusable execution objects or continuity of
  optimizers, schedulers, parameters, or global steps. Changing configuration and
  execution state makes complete checkpoint restoration inherently difficult.
- **Scheduling is driven by the supplied step.** Built-in schedules derive values
  from configuration and that step, without prior-call or accumulated-history
  dependencies. Scheduling does not own global-step continuity. Registered
  schedules and current target values are operational state, distinct from
  training-history state.
- **RunWeaver supplies reusable mechanisms.** Independent components remain
  independently usable. Application-specific training policy and restrictions
  motivated only by generic framework conventions do not belong in these contracts.

## Historical compromise — developer-confirmed origin

`ParamProvider.get()` evolved to combine access to targets, run parameters, and
other configuration dictionaries. Its precedence is established historical
behavior, not necessarily an idealized design. Changing it may alter existing
experiments and requires deliberate consideration; it is neither automatically
defective nor an immutable compatibility guarantee.

**Implementation observation:** lookup checks targets first; then top-level keys
in `run`, `global_cfg`, and `runtime`; then keys in immediate child dictionaries
of those same layers, in that order. Nested collisions use the first matching
child in dictionary insertion order. Thus any top-level match precedes every
nested match. Lookup is one level deep. Missing `get()` values return `None`;
`has()` distinguishes absence from a present `None`. An `overrides` dictionary
has no special priority or temporary-restoration semantics.
Source: [ParamProvider](../src/runweaver_ml/phase_control/param_provider.py#L25).

## Actionable implementation defect — source-derived

**Partial schedule registration:** `ScheduleEngine.register()` records a block
before evaluating its initial value. If evaluation fails, the block remains
registered while the target retains its previous value. A corrected retry is
then rejected as a duplicate. The smallest proposed correction is to evaluate
successfully before committing registration. No implementation fix is included
in this record.
Source: [registration](../src/runweaver_ml/phase_control/scheduling/schedule_engine.py#L20).

## Existing contracts — source-derived, documentation status confirmed

- **Restoration:** `state_dict()` stores only `last_step`, not definitions or
  target values. `load_state_dict()` loads that step without recalculating already
  registered targets; a subsequent `update(step)` synchronizes them. Reconstructing
  built-in schedules and evaluating at the appropriate phase-local step requires
  no earlier evaluations. This is a contract to document, not an architectural
  defect. Source: [engine state methods](../src/runweaver_ml/phase_control/scheduling/schedule_engine.py#L51).
- **Equal boundaries:** linear and cosine schedules with
  `start_step == end_step` return `start` at and before that boundary, and `end`
  afterward. Do not change the formulas or classify this as defective without a
  demonstrated requirement. Sources: [linear](../src/runweaver_ml/phase_control/scheduling/schedules/schedule_classes.py#L43)
  and [cosine](../src/runweaver_ml/phase_control/scheduling/schedules/schedule_classes.py#L74).

## Open engineering question

**Validation failure cleanup:** training protects its end hook with `finally`
after the begin hook, whereas validation does not guarantee its end hook or
collector finalization after failure. Whether validation should guarantee cleanup
remains open for future consideration; this does not invalidate the architecture.
Sources: [training and validation loops](../src/runweaver_ml/phase_control/phase_orchestrator.py#L24)
and [validation finalization](../src/runweaver_ml/phase_control/execution/phase_training_module.py#L133).

## Documentation maintenance

These are documentation issues, not architectural defects:

- Explain the complete parameter lookup precedence and collision rules in
  [ARCHITECTURE.md](../ARCHITECTURE.md#runtime-parameters-and-schedules).
- Correct cosine and step type names and field examples; clarify that square-wave
  `duty` is a fraction (`0.5` means 50%) in the
  [schedule comments](../src/runweaver_ml/phase_control/scheduling/schedules/schedule_classes.py#L61).
- Remove obsolete claims about a parameter override context manager, a compiler
  returning a dataclass, and manager indexing in
  [phase-description comments](../src/runweaver_ml/phase_control/train_phase_manager.py#L3).
  The compiler returns a dictionary and the manager exposes iteration, not an
  indexing method; [ParameterWrapper](../src/runweaver_ml/phase_control/param_wrapper.py#L12)
  has no override context manager.

## Review Principles

1. Consult `ARCHITECTURE.md` and this record before evaluating components.
2. Distinguish developer-confirmed intentions from implementation observations.
3. Respect intentional flexibility and historical compromises.
4. Identify concrete consequences before recommending changes.
5. Prefer minimal corrections that preserve existing design intentions.
6. Do not repeatedly report acknowledged tradeoffs as defects.
7. Ask for clarification when documentation and source cannot establish intent.

A different design being possible is not evidence that the existing design is wrong.
