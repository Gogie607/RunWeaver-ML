# DSM transport extensions

The public imports are `TransportBase`, `TransportManifest`, `DeferredPayload`,
and `MemoryTransport` from `runweaver_ml.dataset_management`.

Implement `TransportBase.iter_samples(*, deliverables, load_policy,
shardshuffle=False, anchor_modality=None)`, and initialize `manifest` before
supplying the object to `MultimodalDataset`. Each call starts a fresh iteration.
The transport owns storage access, synchronization, its sample universe, and
resource lifetime. Open and close thread-bound resources within iteration;
the root is consumed by the prefetch fetch thread.

`manifest.available_modalities` maps normalized modality names to serializer
identifiers. `None` declares native Python values; other identifiers select
`FILETYPE_LOADERS` codecs in the existing layout module. Publish the complete
mapping before construction. MMDS and LayoutSource take dictionary snapshots.
The manifest's fragment and shard lists remain available for TAR discovery;
alternative transports need not invent filesystem fragments.

Each emitted dictionary requires `__key__` (stable sample identity) and contains
requested modality values. `__shard__` is optional partition identity: ordinary
transports, providers, views, and batching do not require it. A supplied partition
need not be a filename. LayoutSource maps `__key__` to `sample_id`, adds the dataset
domain, and preserves `__shard__` only when supplied. Reserved fields beginning with `__` are not modalities. Requested fields
needed by providers must be included in the root's construction-time contract.
Missing per-sample fields retain the existing optional-field behavior.

Honor partition shuffle and anchor selection, or reject unsupported options
explicitly. The existing `op_params.layout.shuffle` performs buffered shuffle
within contiguous shard groups, and `shardshuffle` requests partition-order
shuffle. These explicitly shard-aware operations require a non-None `__shard__`
on every selected sample and fail clearly if it is missing. No shard is invented
for an unpartitioned source. Partitions should be contiguous for buffered shuffle.
`load_policy` lists fields eligible for deferred loading; eager delivery remains
valid. `DeferredPayload(callable)` delays payload access until root decoding;
it is resolved before provider filtering, including native nullable values.
Successful and failed loads release their loader references. Failed loads are
terminal; a fresh iteration supplies a new payload for retry. `payload.close()`
explicitly abandons an unloaded payload; a successfully cached value remains
usable after close(). TAR deferred payloads share an indexed reader with explicit
reference-counted ownership. Each load uses its indexed TarInfo directly; it does
not rescan member names or reopen the archive. Access is serialized to protect
shared seek/decompression state during concurrent prefetch. The reader closes
when the last outstanding loader is loaded, fails, is explicitly closed, or is
collected. Iterator shutdown abandons only samples that were not delivered;
delivered payloads remain usable afterward. Retaining unloaded payloads therefore
intentionally retains their reader. Source contents must remain unchanged until
loads complete. Compressed TAR still incurs decompression when seeking backward,
as in the previous implementation.

Optionally expose `num_samples` (a nonnegative integer for the default universe).
MMDS also accepts an explicit `num_samples` override. With neither count, MMDS
counts raw transport samples without decoding or resolving deferred payloads.
An explicit anchor triggers a fresh raw count instead of using the default
transport count, unless the caller overrides the count. Counts are before
provider filtering. Counting requires reiteration and can be expensive.
Default root-based TAR metadata generation preserves its first-discovered-modality
count universe, but also counts raw samples with deferred loading. It releases
unneeded loaders without decoding. Existing metadata caches and their JSON format
remain unchanged; counting is not validation of serialized payload contents.

## Supplying a transport

```python
from runweaver_ml.dataset_management import (
    MemoryTransport, MultimodalDataset, create_loader,
)

transport = MemoryTransport([
    {"__key__": 101, "text": "hello", "label": 1},
    {"__key__": 102, "text": "world", "label": 2},
])
dataset = MultimodalDataset(transport=transport, domain="example")
# dataset.install_provider(provider) uses the existing processing pipeline.
loader = create_loader(dataset, ["text", "label"], batch_size=2)
batch = next(iter(loader))
```

Supply a contributor-defined transport the same way. No registry or internal
modification is needed. `build_dataset` forwards `transport` and `num_samples`
(pass `root=None`); `build_datasets` accepts those values in each Python dataset
configuration dictionary. Transport objects are not YAML backend selectors.
An explicit domain is required for supplied transports. Root is optional and
does not drive discovery or metadata caching for supplied transports.

MemoryTransport stores native values, preserves optional supplied partitions,
projects requested fields, supports partition shuffle when all selected samples
have partitions, supports sparse anchors, and creates fresh top-level dictionaries
on reiteration. It never fabricates a logical shard; shardless manifests have an
empty shard list. Shardless inputs retain input order; fully partitioned inputs
retain the existing grouping by partition, with optional partition-order shuffle.
Nested mutable payload objects remain caller-owned; it is a reference transport,
not a deep-copy or large-scale buffering facility.

## Shard-aware providers and collation

A persistent or otherwise shard-aware ProviderBase declares `__shard__` in its
existing `requires` list, for example `requires = ["text", "__shard__"]`.
MMDS validates modality dependencies during installation; optional shard metadata
is checked per sample by ProviderBase.iterate, before process() runs. This supports
mixed or filtered streams without claiming all transports provide shards.
Custom middleware that implements its own iterate() must enforce its own
shard-dependent preconditions. A provider with only a conditional shard-dependent
branch should validate identity in that branch instead of requiring it globally.
DatasetView passes optional metadata through without adding or requiring it.
RunWeaver does not ship a built-in persistent provider.

DynamicCollator omits `__shard__` when no row supplies a non-None value. When at
least one row supplies it, the batch retains an aligned list of those values,
with None marking absent identities in mixed batches. It does not create shard
names or turn missing identities into string placeholders, even when `__shard__`
is listed among requested modalities. Fully sharded TAR batches keep their
existing shard-name lists.

## Compatibility and verification

Existing `MultimodalDataset(root, ...)` and root-based builder configurations
still construct TarTransport, preserve local metadata JSON caching, and use the
existing TAR discovery, normalization, synchronization, codecs, and lazy policy.
Directory validation now belongs to TarTransport.

DatasetLayout and its export have been removed. Low-level LayoutSource callers
now supply `transport` and `domain`; callers of `dataset.layout` use
`dataset.transport`/`dataset.available_modalities` instead. Storage-aware
providers that formerly used `view.root.layout.root` must use their concrete
transport's root. That root is not part of the generic transport contract.
The LI application's WAV-duration fallback has been updated accordingly.

`tests/test_transport_extensions.py` exercises native and serialized/deferred
delivery, nulls, provider filtering/artifacts, prefetch, DataLoader, reiteration,
raw counting, builder/manager composition with TAR, and invalid contracts. It
also covers shardless end-to-end delivery, mixed collation, explicit partition
requirements, and representative application persistence into a TAR shard.
Its test-local ExternalTransport demonstrates a contributor implementation using
the public interface. Existing TAR boundary tests continue exercising decoding,
synchronization, sparse anchors, lazy loading, and shuffle.

SQL is not implemented. Connection safety, query consistency and cancellation
remain transport responsibilities. Existing prefetch workers can reorder sample
delivery. TAR's local WebDataset streams are explicitly owned and closed even
when exception objects are retained. Prefetch closes its active root iterator
in its fetch-thread finally block, and queue-waiting workers respond to shutdown.
Blocking storage reads still must return before that thread can finalize; the
existing bounded thread joins are retained. These lifetime changes do not
redesign sample identity, provider negotiation, or metadata caching.
