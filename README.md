# Parquet streaming merge example

Observed across 5 direct `--release` runs on an M3 Max MBP:

- `streaming_merge`: `115-162µs`
- `async_streaming_merge`: `161-206µs`
- `async_streaming_merge_widening`: `205-348µs`
- `async_streaming_merge_payload`: `420-822µs`

These are the in-program merge timings printed by the examples themselves, not total `cargo run` wall-clock time.

## Async widening merge example

There is now a widening-focused async merge example at `examples/async_streaming_merge_widening.rs`.

Run it with:

`cargo run --example async_streaming_merge_widening`

Run the library and payload tests with:

`cargo test`

### Widening behavior

- Primitive widening is permissive: numeric types are promoted to a common type and primitive/string combinations can promote to string.
- Structured widening is shape-aware: `Struct + Struct` unions fields by name, widens shared children recursively, and makes missing fields nullable.
- List widening is supported for `List` and `LargeList`, including mixed combinations where the merged type is promoted to `LargeList` when needed.
- Primitive vs structured conflicts remain strict by default (for example `Utf8` vs `Struct` returns an incompatibility error).

## Payload-native typed merge

The preferred contract in this repo is:

- stable envelope columns at the top level
- one evolving `payload: Struct<...>` column for queryable business properties

The payload-aware example lives at `examples/async_streaming_merge_payload.rs`.

Run it with:

`cargo run --example async_streaming_merge_payload`

### Payload contract

- Every input file must already contain a top-level `payload` column.
- `payload` must be an Arrow `Struct`.
- Non-payload top-level columns are treated as the stable envelope and must match by name and datatype across inputs, though order may differ and nullability may widen.
- Only the `payload` subtree is widened recursively.
- Incompatible payload shapes still fail fast; this repo does not use Parquet `VARIANT` as the query contract.

### Compiled Engine

The payload path now uses a compiled compaction engine instead of rebuilding name lookups inside the per-batch loop:

- planning computes the output schema once and compiles per-source adapters
- execution streams batches through cached adapters with an identical-schema fast path
- nested `Struct` and `List` coercion uses precomputed child indices instead of repeated string lookups

It also now supports an ordered Parquet merge path:

- `merge_payload_parquet_files_with_execution(...)` accepts `ParquetMergeExecutionOptions`
- ordered mode performs a true k-way merge over inputs already sorted by one top-level envelope column
- output ordering is ascending with `NULLS LAST` and stable tie-breaking by input-file order
- supported ordered-merge key types are `Int64`, `UInt64`, `Float64`, `Utf8`, `LargeUtf8`, `Date32`, `Date64`, and `Timestamp(_, _)`
- ordered mode validates that each input file is itself sorted on the chosen envelope column

The same engine also powers a two-pass NDJSON compaction flow:

- pass 1 discovers the exact typed payload union schema across all NDJSON inputs
- pass 2 streams NDJSON into Arrow builders and writes Parquet without buffering the full dataset in memory
- optional in-memory sorting can reorder NDJSON output by one envelope field before Parquet write

### NDJSON sorting

- `CompactionOptions.sort_field` enables ascending stable sort with `NULLS LAST` before Parquet write.
- The sort field must also be listed in `envelope_fields`.
- Supported sort-key types are `Int64`, `UInt64`, `Float64`, `Utf8`, and `LargeUtf8`.
- Sorting uses the typed Arrow output columns, not raw JSON text.
- `CompactionOptions.sort_max_rows_in_memory` defaults to `262_144`; jobs above that cap fail clearly because external spill sort is not implemented yet.

### Query proof

The query-engine proof in this repo is DataFusion nested field access over typed structs, for example:

`SELECT sum(payload['score']) FROM merged_payload`

That keeps nested payload properties first-class and typed in the query engine without depending on Parquet `VARIANT` accessors.

## Benchmarking

Run the compiled payload benchmark with:

`cargo run --release --example payload_benchmark`

It reports:

- payload-native Parquet merge throughput
- NDJSON compaction throughput for mixed-drift, sorted mixed-drift, repeated-shape, and unique-shape workloads
- compaction-plus-merge throughput
- planning time, execution time, sorting time, rows/sec, input MB/sec, and peak RSS
- planning threads used, unique shapes discovered, and shape-cache hit/miss counts
- peak RSS is process high-water mark, so later benchmark lines reflect the highest peak reached anywhere in the run

Run the dedicated Parquet merge benchmark with:

`cargo run --release --example parquet_merge_benchmark`

It reports:

- unordered payload merge baseline throughput
- ordered k-way merge throughput for identical-schema and alternating-schema payload inputs
- ordered merge throughput for a few huge files with many row groups
- ordered merge throughput for many smaller sorted files
- planning time, execution time, ordered-merge time, rows/sec, input MB/sec, input/output batch counts, adapter cache hits/misses, and peak RSS
