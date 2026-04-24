# Parquet streaming merge example

## Async widening merge example

The widening-focused async merge example lives at `examples/async_streaming_merge_widening.rs`.

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

The payload path uses a compiled compaction engine instead of rebuilding name lookups inside the per-batch loop:

- planning computes the output schema once and compiles per-source adapters
- execution streams batches through cached adapters with an identical-schema fast path
- nested `Struct` and `List` coercion uses precomputed child indices instead of repeated string lookups

It also supports an ordered Parquet merge path:

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

Run the Rust-vs-DuckDB comparison benchmark with:

`cargo run --release --example rust_vs_duckdb_benchmark`

It reports:

- one strict top-level merge workload using the new top-level merge path
- one nested payload merge workload using the payload-aware merge path
- one ordered payload merge workload using the ordered k-way merge path
- side-by-side Rust and DuckDB timings for the exact same Parquet inputs
- row-count and merged-schema validation before timings are accepted
- a JSON summary written into the benchmark artifact directory

Useful env vars:

- `RPM_BENCH_SCENARIO=top_level_pragmatic`, `nested_payload_pragmatic`, `ordered_payload_pragmatic`, or `all`
- `RPM_BENCH_TARGET_INPUT_GIB=<float>` to scale generated input size by approximate total GiB
- `RPM_BENCH_FILE_COUNT=<int>` to change the number of input files
- `RPM_BENCH_MEASURED_RUNS=<int>` to reduce or increase measured repetitions
- `RPM_BENCH_RUST_PARALLELISM=<int>` to tune Rust merge parallelism (`1` is low-resource/deterministic default, `0` auto-caps to available cores and input count)
- `RPM_BENCH_RUST_UNORDERED_ORDER=preserve` or `interleaved` to preserve input-file row order or allow fastest unordered writes
- `RPM_BENCH_RUST_COMPRESSION=uncompressed`, `snappy`, `lz4_raw`, or `zstd:<level>` to choose Rust output compression
- `RPM_BENCH_RUST_DICTIONARY=true` or `false` to toggle Rust dictionary encoding
- `RPM_BENCH_RUST_READ_BATCH_SIZE=<int>` to tune Rust Parquet read batch size
- `RPM_BENCH_RUST_OUTPUT_BATCH_ROWS=<int>` to tune Rust output batch materialization size
- `RPM_BENCH_RUST_OUTPUT_ROW_GROUP_ROWS=<int>` to tune Rust output row group size
- `RPM_BENCH_RUST_PREFETCH_BATCHES_PER_SOURCE=<int>` to tune per-source buffering
- `RPM_BENCH_EXACT_VALIDATION=true` to add DuckDB `EXCEPT ALL` validation both ways
- `RPM_BENCH_DUCKDB_THREADS=<int>` to set DuckDB `PRAGMA threads`
- `RPM_BENCH_DUCKDB_COMPRESSION=snappy`, `uncompressed`, `zstd`, or `lz4` to set DuckDB output compression

### 1 GiB top-level snapshot

Environment: M3 Max MBP, DuckDB CLI `v1.4.1`, `--release`, one measured run.

`RPM_BENCH_SCENARIO=top_level_pragmatic RPM_BENCH_TARGET_INPUT_GIB=1 RPM_BENCH_MEASURED_RUNS=1 RPM_BENCH_RUST_PARALLELISM=0 RPM_BENCH_RUST_COMPRESSION=snappy cargo run --release --example rust_vs_duckdb_benchmark`

- total input: `1031.70 MiB`
- Rust resolved parallelism: `6`
- Rust `1161 ms`, output `646,169,747` bytes, Snappy
- Rust peak RSS `861.78 MiB`; CPU `7288 ms` total (`5598 ms` user, `1689 ms` sys, `628%` of wall)
- DuckDB `1153 ms`, output `455,286,989` bytes, Snappy
- DuckDB peak RSS `2954.08 MiB`; CPU `6856 ms` total (`5872 ms` user, `983 ms` sys, `595%` of wall)
- Rust internal breakdown: decode `794 ms`, prepare `104 ms`, writer elapsed `1155 ms`, encode work `5882 ms`, sink `437 ms`
- Output validation: row count and schema matched DuckDB (`55,202,568` rows each)
- Result: Rust and DuckDB are effectively tied on this top-level 1 GiB single-file workload when using DuckDB-comparable Snappy output.

Compression matrix for the same snapshot, all with `RPM_BENCH_RUST_PARALLELISM=0`. Deltas are Rust minus DuckDB, so negative RSS means lower Rust peak memory and positive CPU means higher Rust total CPU:

| Rust compression | Rust median | Rust output bytes | Rust peak RSS | Rust CPU | DuckDB median | DuckDB output bytes | DuckDB peak RSS | DuckDB CPU | RSS delta | CPU delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `snappy` | `1161 ms` | `646,169,747` | `861.78 MiB` | `7288 ms` (`628%`) | `1153 ms` | `455,286,989` | `2954.08 MiB` | `6856 ms` (`595%`) | `-2092.30 MiB` | `+432 ms` |
| `zstd:1` | `1659 ms` | `390,729,708` | `955.00 MiB` | `9708 ms` (`585%`) | `1254 ms` | `455,286,989` | `2945.66 MiB` | `6616 ms` (`527%`) | `-1990.66 MiB` | `+3092 ms` |

Snappy is the DuckDB-comparable Rust benchmark setting because it matches DuckDB's default Parquet output compression. Zstd level 1 is also competitive when output size matters: in this snapshot it produces an output file about 40% smaller than Rust Snappy, with higher CPU and wall time.

The benchmark JSON includes cumulative row-group encode worker time. That value is summed across parallel workers, so it is expected to be larger than elapsed writer wall time.

### 1 GiB ordered payload snapshot

`RPM_BENCH_SCENARIO=ordered_payload_pragmatic RPM_BENCH_TARGET_INPUT_GIB=1 RPM_BENCH_MEASURED_RUNS=1 RPM_BENCH_RUST_PARALLELISM=0 RPM_BENCH_RUST_COMPRESSION=snappy cargo run --release --example rust_vs_duckdb_benchmark`

- total input: `941.89 MiB`
- Rust resolved parallelism: `6`
- Rust `5524 ms`, output `366,706,298` bytes, Snappy
- Rust peak RSS `2157.50 MiB`; CPU `11439 ms` total (`10386 ms` user, `1053 ms` sys, `207%` of wall)
- DuckDB `1079 ms`, output `315,281,804` bytes, Snappy
- DuckDB peak RSS `4943.62 MiB`; CPU `9458 ms` total (`8153 ms` user, `1304 ms` sys, `876%` of wall)
- Delta: Rust `+4445 ms` wall, `-2786.12 MiB` peak RSS, `+1981 ms` total CPU
- Rust ordered breakdown: decode `1313 ms`, prepare `209 ms`, assembly `3443 ms` (`1569 ms` selection, `1873 ms` materialization), writer elapsed `5469 ms`, encode work `4299 ms`, sink `175 ms`
- Ordered output used `186` interleave flushes, `0` concat flushes, and `0` direct writes for the dense row-interleaved workload.

Important: the comparison should be run in `--release`. A debug-mode `cargo run` makes the Rust merge path look artificially slow and is not a fair comparison against the optimized DuckDB CLI binary.

For a much larger startup-latency-insensitive run, a 10 GiB target is:

`RPM_BENCH_SCENARIO=top_level_pragmatic RPM_BENCH_TARGET_INPUT_GIB=10 RPM_BENCH_MEASURED_RUNS=1 cargo run --release --example rust_vs_duckdb_benchmark`
