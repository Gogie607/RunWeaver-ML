# DSM Artifact Implementation Review

Status: analysis and recommendations only; no implementation changes.
The developer's clarified contract is authoritative: artifacts are named value
sources, resolved without arguments, with separate raw-object access. They support
providers and can also supply explicitly selected layout fields. Layout selection
and provider resource dependencies are distinct contracts.

## Current architecture and lifecycle

| Component | Implemented responsibility |
| --- | --- |
| `Artifact` | Holds a name and object; `value(*args, **kwargs)` invokes callable objects, otherwise returns them; `item()` returns the stored object. |
| `BaseArtifact` | Adds subclass registration, an abstract dataset-aware `from_config()`, and an unused DSM `requires` declaration. |
| `ArtifactStore` | Registers wrappers by name, enforces uniqueness, returns wrappers, and supports membership/removal. It does not resolve values itself. |
| `create_artifacts()` | Resolves application-defined classes by handler/name and invokes construction once per dataset. |
| Dataset/manager | Registers artifacts before provider installation; distributes shared references or per-dataset factory results; aggregates raw contents. |
| Provider | Declares `artifact_requires`, accesses resources through `view.artifacts`, and processes samples. |

Data flow: application definitions/configuration -> `build_datasets()` ->
`create_artifacts(config, dataset_config)` -> `build_dataset()` -> dataset store ->
provider dependency validation -> provider processing. Alternatively, an existing
manager's `distribute_artifact()` registers resources on each child dataset; dependent
providers must be installed afterward. Registration is eager; stored callables are
evaluated only when explicitly resolved (or injected by the layout layer).

Sources: [Artifact](src/runweaver_ml/dataset_management/artifacts/artifact.py#L12),
[BaseArtifact/factory](src/runweaver_ml/dataset_management/artifacts/artifact_base.py#L12),
[store](src/runweaver_ml/dataset_management/artifacts/artifact_store.py#L27),
[builder](src/runweaver_ml/dataset_management/build_dataloader.py#L23),
[distribution](src/runweaver_ml/dataset_management/engine/multi_dataset_manager.py#L253),
[provider installation](src/runweaver_ml/dataset_management/engine/multimodal_dataset.py#L132).

## Specific inconsistencies

1. **Resolution belongs to the wrapper rather than the store.**
   `ArtifactStore.get()` and `__getitem__()` return wrappers; store-level `item(name)`
   is absent. Consumers must know to call `.value()` or `.item()` afterward.
   [Lookup](src/runweaver_ml/dataset_management/artifacts/artifact_store.py#L48).
2. **Resolution forwards arbitrary arguments.** `Artifact.value(*args, **kwargs)`
   permits sample/model-specific invocation through the artifact mechanism.
   Resolution must instead call a value source with no arguments. Callable models
   remain legal stored objects: providers retrieve them through raw-object access
   and invoke them explicitly as processing work. Do not infer signatures or catch
   invocation errors to silently return the object.
   [Evaluation](src/runweaver_ml/dataset_management/artifacts/artifact.py#L51).
3. **Correction: artifact-backed layout selection is supported behavior, not an
   inconsistency.** `LayoutSource` adds registered artifact names to capabilities
   and resolves only selected fields into samples, taking precedence over
   same-named transport fields. This assembles non-sample-aware values into the
   stream; it does not make artifacts sample-processing providers. Preserve it.
   With explicit `requires`, unrequested artifacts are neither inserted nor
   evaluated. `MultimodalDataset(requires=None)` defaults to transport names;
   directly constructing `LayoutSource(deliverables=None)` selects all capabilities,
   including artifacts. Provider `artifact_requires` checks resource presence and
   neither changes layout selection nor automatically creates missing resources.
   [Capabilities](src/runweaver_ml/dataset_management/engine/layout/dataset_layout_source.py#L53),
   [injection](src/runweaver_ml/dataset_management/engine/layout/dataset_layout_source.py#L96).
4. **Descriptions and helper interfaces retain the broader conceptual model.**
   `Artifact` describes self-operationalization and sample insertion; these are
   not its implemented operations. `BaseArtifact.requires` has no enforcement in
   DSM. Dataset helpers duplicate wrapper/value/raw access, and value/raw helpers
   incorrectly annotate their results as `Artifact`. Provider documentation names
   `view.artifact_store`, whereas actual access is `view.artifacts`.
   [Artifact description](src/runweaver_ml/dataset_management/artifacts/artifact.py#L14),
   [helpers](src/runweaver_ml/dataset_management/engine/multimodal_dataset.py#L178),
   [provider contract](src/runweaver_ml/dataset_management/source/provider_base.py#L12).
5. **Distribution conflates omission with a constant `None`.** `value=None` is
   rejected even though a store can hold `None` inside a wrapper. Use an omitted-value
   sentinel when implementing the revised API.
   [Argument check](src/runweaver_ml/dataset_management/engine/multi_dataset_manager.py#L295).

## Recommended smallest coherent architecture

**ArtifactStore can directly provide the runtime contract.** Store ordinary Python
objects by name without copying, wrapping, or pre-evaluating them:

```python
store.register(name, source, overwrite=False)
store.get(name)   # source() if callable(source), otherwise source
store.item(name)  # exact stored object, never evaluated
store.has(name)
store.names()
store.deregister(name, silent=False)
store.clear()
```

Recommended missing-name policy: `get()` and `item()` raise `KeyError`; stored
`None` is valid. This is a proposed decision, not an already agreed contract.
If indexing is retained, make `store[name]` equivalent to `get(name)`; raw access
must remain explicit. Preserve membership, length, and diagnostic representation.
Every `get()` evaluates a callable anew: no caching or state reset. A model requiring
inputs uses `item()`; a parameterless source returning that model can use `get()`.

**Eliminate `Artifact` as a required wrapper.** Its name belongs to the store key;
its resolution/raw access can move directly to the store. No unique runtime
capability requires wrapper identity. Stored resource identity must survive.

**Remove the inheritance requirement represented by `BaseArtifact`.** Configuration
construction remains useful, but need not be an artifact subclass hierarchy.
Reuse the existing handler registry as a mapping to application-owned construction
callables with the existing two-input factory contract:
`factory(artifact_config, dataset_config) -> source`. Preserve the established
`name`/`handler`/`config` selection format; have `create_artifacts()` return named
sources for registration. Existing class-name selector aliases require an explicit
migration inventory. Do not introduce another registry or resource framework.
Existing application factories can replace subclass `from_config()` methods.

**Keep distribution in the manager.** Preserve `value=source` as shared-reference
distribution and `create(dataset)` as once-per-dataset construction. Calling a
dataset-aware construction factory is setup, not argument forwarding during value
resolution. Aggregation should use `store.item(name)` and fail clearly for missing
names. Providers retain `artifact_requires`; presence checks must not evaluate
resources. Preserve layout capability resolution across transport fields and
registered artifacts, and parameterless resolution of selected artifact fields.
Provider resource declarations must not imply field insertion or evaluation.
Layout capabilities are captured at construction; later artifact distribution
does not automatically extend that fixed selection contract.

## Functionality that must survive

- Shared resource identity and per-dataset independent construction, including
  multiple datasets using one transport with different domains.
- Stateful parameterless callables, whose state progresses on resolution.
- Arbitrary raw objects, including RNGs and models, with explicit unevaluated access.
- Configuration-driven setup before provider installation, dependency validation,
  declaration order, and unchanged transport/provider separation.
- Artifact-backed layout fields when selected, current collision precedence,
  and the separation between field selection and provider resource requirements.
- Application ownership of concrete resource creation/configuration; no framework
  application-specific sample processing or unrelated application-state store.
  Callable resources requiring inputs must use raw-object access, not be selected
  as parameterless artifact-backed fields.

## RuntimeArtifactRepository boundary

This repository separately creates configured resources lazily and caches their
instances for its repository lifetime. The dataset store resolves value sources
on each access and preserves distributed identity. Both associate names with
objects, but their construction/evaluation and lifetimes differ. **Do not merge them.**
Its handler/config construction is not a violation of parameterless dataset value
resolution. [Repository implementation](src/runweaver_ml/dataset_management/runtime_artifact_repository.py#L3).

V2's `TrainingSession._build_artifact_repository()` currently supplies
`ARTIFACT_REGISTRY` to this repository; V1 also defines application artifact subclasses.
Changing registry entry signatures must not silently change repository factories.
Migrate those application registrations deliberately, or retain the existing
construction-only class interface temporarily while direct store semantics are adopted.
This is a migration constraint, not a reason to retain runtime wrappers indefinitely.

## Affected examples, tests, and documentation

- [Transport extension tests](tests/test_transport_extensions.py#L13): replace
  `Artifact("suffix", "!")` construction and `.get(...).value()` provider access;
  preserve provider output/filtering and reiteration assertions.
- [Transport/layout tests](tests/test_tar_transport_layout_boundary.py#L8): empty
  stores remain usable; preserve layout selection and decoding behavior.
- [Memory provider example](scripts/demo_memory_provider.py#L18): replace the
  `BaseArtifact` factory and wrapper lookup with named source construction/raw access.
- [Identity example](scripts/demo_artifact_identity.py#L20): report store/resource
  identities rather than obsolete wrapper IDs; retain shared/independent state checks.
- [DSM guide](src/runweaver_ml/dataset_management/docs/README.md#L85): preserve
  documented artifact-backed `requires` behavior. Clarify defaults, collision
  precedence, and resource dependencies versus sample selection. Update architecture,
  exports, README terminology, and external application consumers consistently.

Current tests do not directly establish the clarified store API, parameterless
callable behavior, or callable-model raw access. Future implementation verification
should cover those contracts, missing versus stored `None`, identity/state progression,
independent factories, dependency checks without evaluation, and migration of
wrapper access while preserving artifact-backed sample fields. Equal output
sequences alone do not prove sharing.

## Minimal migration plan

1. Agree on missing-name/indexing semantics and inventory application wrapper,
   subclass, registry, and artifact-modality consumers before changing interfaces.
2. Implement direct store registration, resolution, and raw access; update dataset
   initialization/helpers, manager distribution/aggregation, and providers together.
3. Adapt existing configuration construction to named sources and migrate application
   factories. Keep RuntimeArtifactRepository's contract independent.
4. Keep artifact-backed layout insertion and route its resolution through the
   simplified store. Remove obsolete wrappers, subclass requirements, comments,
   and exports after affected uses migrate; do not require replacement providers
   for layout-selected artifacts.
5. Update examples/documentation and verify preserved behavior with focused tests.

This is an intentional public-interface change requiring approval before implementation;
no compatibility layer is recommended without a demonstrated migration need.

## Verification

Source and consumer review only; no implementation changed. The existing 46-test
RunWeaver suite passed during this review. Those tests validate current behavior,
not the proposed API. Example identity/state checks were executed in the preceding
challenge; they were inspected here, not rerun.

Follow-up layout verification: in-memory checks confirmed requested artifact
insertion without providers, exclusion of unrequested artifacts, artifact-over-
transport collision precedence, both default-selection paths, and provider
dependency validation without evaluation, insertion, or automatic construction.
Existing tests exercise provider access to artifacts, but no direct automated
artifact-backed layout-selection/collision test was found. The DSM guide explicitly
documents artifact values in `requires`; its language/speaker examples are not
evidence that any particular training objective consumes those fields.
