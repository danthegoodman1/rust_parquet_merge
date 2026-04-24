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
- ordered payload merge workloads using the ordered k-way merge path, including dense interleaving and mixed row-group-copy shapes
- side-by-side Rust and DuckDB timings for the exact same Parquet inputs
- row-count and merged-schema validation before timings are accepted
- a JSON summary written into the benchmark artifact directory

Useful env vars:

- `RPM_BENCH_SCENARIO=top_level_pragmatic`, `nested_payload_pragmatic`, `ordered_payload_pragmatic`, `ordered_payload_mixed_pragmatic`, or `all`
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
- `RPM_BENCH_RUST_ORDERED_MEMORY_BUDGET_MIB=<int>` to opt ordered merge into the higher-throughput byte-bounded pipeline; `4096` is the current speed target
- `RPM_BENCH_EXACT_VALIDATION=true` to add DuckDB `EXCEPT ALL` validation both ways
- `RPM_BENCH_DUCKDB_THREADS=<int>` to set DuckDB `PRAGMA threads`
- `RPM_BENCH_DUCKDB_COMPRESSION=snappy`, `uncompressed`, `zstd`, or `lz4` to set DuckDB output compression

### 1 GiB top-level snapshot

Environment: M3 Max MBP, DuckDB CLI `v1.4.1`, `--release`, one measured run.

`RPM_BENCH_SCENARIO=top_level_pragmatic RPM_BENCH_TARGET_INPUT_GIB=1 RPM_BENCH_MEASURED_RUNS=1 RPM_BENCH_RUST_PARALLELISM=0 RPM_BENCH_RUST_COMPRESSION=snappy cargo run --release --example rust_vs_duckdb_benchmark`

- total input: `1031.70 MiB`
- Rust resolved parallelism: `6`
- Rust `1217 ms`, output `646,169,747` bytes, Snappy
- Rust peak RSS `771.00 MiB`; CPU `7583 ms` total (`5738 ms` user, `1844 ms` sys, `623%` of wall)
- DuckDB `1187 ms`, output `455,286,989` bytes, Snappy
- DuckDB peak RSS `2912.62 MiB`; CPU `7171 ms` total (`6043 ms` user, `1128 ms` sys, `604%` of wall)
- Output validation: row count and schema matched DuckDB (`55,202,568` rows each)
- Result: Rust and DuckDB are effectively tied on this top-level 1 GiB single-file workload when using DuckDB-comparable Snappy output.

Compression matrix for the same snapshot, all with `RPM_BENCH_RUST_PARALLELISM=0`. Deltas are Rust minus DuckDB, so negative RSS means lower Rust peak memory and positive CPU means higher Rust total CPU:

| Rust compression | Rust median | Rust output bytes | Rust peak RSS | Rust CPU | DuckDB median | DuckDB output bytes | DuckDB peak RSS | DuckDB CPU | RSS delta | CPU delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `snappy` | `1217 ms` | `646,169,747` | `771.00 MiB` | `7583 ms` (`623%`) | `1187 ms` | `455,286,989` | `2912.62 MiB` | `7171 ms` (`604%`) | `-2141.62 MiB` | `+412 ms` |
| `zstd:1` | `1345 ms` | `390,729,708` | `845.53 MiB` | `8745 ms` (`650%`) | `1184 ms` | `455,286,989` | `3381.31 MiB` | `7191 ms` (`607%`) | `-2535.78 MiB` | `+1554 ms` |

Snappy is the DuckDB-comparable Rust benchmark setting because it matches DuckDB's default Parquet output compression. Zstd level 1 is also competitive when output size matters: in this snapshot it produces an output file about 40% smaller than Rust Snappy, with higher CPU and wall time.

The benchmark JSON includes cumulative row-group encode worker time and ordered materialization work. Those values are summed across parallel workers, so they can be larger than the corresponding elapsed wall time.

### 1 GiB ordered payload snapshots

Environment: M3 Max MBP, DuckDB CLI `v1.4.1`, `--release`, one measured run, Rust Snappy output, `RPM_BENCH_RUST_PARALLELISM=0`, and `RPM_BENCH_RUST_ORDERED_MEMORY_BUDGET_MIB=4096`.

Mixed ordered target:

`RPM_BENCH_SCENARIO=ordered_payload_mixed_pragmatic RPM_BENCH_TARGET_INPUT_GIB=1 RPM_BENCH_MEASURED_RUNS=1 RPM_BENCH_RUST_PARALLELISM=0 RPM_BENCH_RUST_COMPRESSION=snappy RPM_BENCH_RUST_ORDERED_MEMORY_BUDGET_MIB=4096 RPM_BENCH_EXACT_VALIDATION=true cargo run --release --example rust_vs_duckdb_benchmark`

- total input: `602.26 MiB`; rows: `74,253,696`
- Rust resolved parallelism: `6`
- Rust `745 ms`, output `651,237,782` bytes, Snappy
- Rust peak RSS `1052.20 MiB`; CPU `2545 ms` total (`1859 ms` user, `685 ms` sys, `341%` of wall)
- DuckDB `1212 ms`, output `755,614,694` bytes, Snappy
- DuckDB peak RSS `6292.09 MiB`; CPU `11021 ms` total (`9531 ms` user, `1490 ms` sys, `909%` of wall)
- Delta: Rust `-467 ms` wall, `-5239.89 MiB` peak RSS, `-8476 ms` total CPU
- Rust ordered breakdown: merge `346 ms`, decode `393 ms`, prepare `23 ms`, assembly `284 ms` (`33 ms` selection, `250 ms` materialization work, `0 ms` materialization wait), writer elapsed `743 ms`, encode work `1388 ms`, sink `486 ms`, close `1 ms`, stats fast path `42 ms`
- Copy path: `60` copied row groups, `62,292,672` copied rows, `528,922,842` copied bytes, copy time `391 ms`
- Dense path: `72` partition jobs, `7,425,360` rows, `33 ms` selection work, `246 ms` materialization work, `0` fallbacks
- Buffer peaks: ordered pipeline `1210.00 MiB`, writer `409.68 MiB`
- Exact validation passed with `rust_minus_duckdb=0` and `duckdb_minus_rust=0`.

Dense interleaved stress:

`RPM_BENCH_SCENARIO=ordered_payload_pragmatic RPM_BENCH_TARGET_INPUT_GIB=1 RPM_BENCH_MEASURED_RUNS=1 RPM_BENCH_RUST_PARALLELISM=0 RPM_BENCH_RUST_COMPRESSION=snappy RPM_BENCH_RUST_ORDERED_MEMORY_BUDGET_MIB=4096 RPM_BENCH_EXACT_VALIDATION=true cargo run --release --example rust_vs_duckdb_benchmark`

- total input: `941.89 MiB`; rows: `24,266,406`
- Rust resolved parallelism: `6`
- Rust `894 ms`, output `394,970,517` bytes, Snappy
- Rust peak RSS `3650.72 MiB`; CPU `10526 ms` total (`9130 ms` user, `1396 ms` sys, `1177%` of wall)
- DuckDB `1081 ms`, output `315,281,804` bytes, Snappy
- DuckDB peak RSS `5379.41 MiB`; CPU `9766 ms` total (`8451 ms` user, `1314 ms` sys, `903%` of wall)
- Delta: Rust `-187 ms` wall, `-1728.69 MiB` peak RSS, `+760 ms` total CPU
- Rust ordered breakdown: merge `594 ms`, decode `2333 ms`, prepare `266 ms`, assembly `3269 ms` (`518 ms` selection, `2750 ms` materialization work, `0 ms` materialization wait), writer elapsed `830 ms`, encode work `6807 ms`, sink `407 ms`, close `1 ms`
- Dense path: `186` partition jobs, `24,266,326` rows, `518 ms` selection work, `2705 ms` materialization work, `0` fallbacks
- Dense output used `186` interleave flushes, `16` concat flushes, `0` direct writes, and no row-group copies.
- Buffer peaks: ordered pipeline `2666.00 MiB`, writer `701.50 MiB`
- Exact validation passed with `rust_minus_duckdb=0` and `duckdb_minus_rust=0`.

Important: the comparison should be run in `--release`. A debug-mode `cargo run` makes the Rust merge path look artificially slow and is not a fair comparison against the optimized DuckDB CLI binary.

For a much larger startup-latency-insensitive run, a 10 GiB target is:

`RPM_BENCH_SCENARIO=top_level_pragmatic RPM_BENCH_TARGET_INPUT_GIB=10 RPM_BENCH_MEASURED_RUNS=1 cargo run --release --example rust_vs_duckdb_benchmark`
