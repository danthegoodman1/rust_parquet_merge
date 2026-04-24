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

- `RPM_BENCH_SCENARIO=top_level_pragmatic`, `nested_payload_pragmatic`, `ordered_payload_pragmatic`, `ordered_payload_mixed_pragmatic`, `ordered_payload_string_pragmatic`, `ordered_payload_large_string_pragmatic`, `ordered_payload_string_mixed_pragmatic`, or `all`
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
- Rust `742 ms`, output `651,237,782` bytes, Snappy
- Rust peak RSS `954.69 MiB`; CPU `2499 ms` total (`1811 ms` user, `688 ms` sys, `337%` of wall)
- DuckDB `1240 ms`, output `755,614,753` bytes, Snappy
- DuckDB peak RSS `6425.19 MiB`; CPU `11761 ms` total (`10248 ms` user, `1512 ms` sys, `948%` of wall)
- Delta: Rust `-498 ms` wall, `-5470.50 MiB` peak RSS, `-9262 ms` total CPU
- Rust ordered breakdown: merge `341 ms`, decode `393 ms`, prepare `19 ms`, assembly `289 ms` (`36 ms` selection, `253 ms` materialization work, `0 ms` materialization wait), writer elapsed `739 ms`, encode work `1318 ms`, sink `483 ms`, close `1 ms`, stats fast path `43 ms`
- Copy path: `60` copied row groups, `62,292,672` copied rows, `528,922,842` copied bytes, copy time `385 ms`
- Dense path: `72` partition jobs, `7,425,360` rows, `36 ms` selection work, `248 ms` materialization work, `0` fallbacks
- Buffer peaks: ordered pipeline `1210.00 MiB`, writer `409.68 MiB`
- Exact validation passed with `rust_minus_duckdb=0` and `duckdb_minus_rust=0`.

Dense Int64 interleaved stress:

`RPM_BENCH_SCENARIO=ordered_payload_pragmatic RPM_BENCH_TARGET_INPUT_GIB=1 RPM_BENCH_MEASURED_RUNS=1 RPM_BENCH_RUST_PARALLELISM=0 RPM_BENCH_RUST_COMPRESSION=snappy RPM_BENCH_RUST_ORDERED_MEMORY_BUDGET_MIB=4096 RPM_BENCH_EXACT_VALIDATION=true cargo run --release --example rust_vs_duckdb_benchmark`

- total input: `941.89 MiB`; rows: `24,266,406`
- Rust resolved parallelism: `6`
- Rust `900 ms`, output `394,970,517` bytes, Snappy
- Rust peak RSS `3577.80 MiB`; CPU `11050 ms` total (`9598 ms` user, `1451 ms` sys, `1226%` of wall)
- DuckDB `1057 ms`, output `315,281,804` bytes, Snappy
- DuckDB peak RSS `5241.06 MiB`; CPU `9897 ms` total (`8590 ms` user, `1306 ms` sys, `936%` of wall)
- Delta: Rust `-157 ms` wall, `-1663.26 MiB` peak RSS, `+1153 ms` total CPU
- Rust ordered breakdown: merge `637 ms`, decode `2327 ms`, prepare `298 ms`, assembly `3590 ms` (`640 ms` selection, `2949 ms` materialization work, `2 ms` materialization wait), writer elapsed `804 ms`, encode work `7389 ms`, sink `439 ms`, close `1 ms`
- Dense path: `186` partition jobs, `24,266,326` rows, `640 ms` selection work, `2927 ms` materialization work, `0` fallbacks
- Dense output used `186` interleave flushes, `16` concat flushes, `0` direct writes, and no row-group copies.
- Buffer peaks: ordered pipeline `2458.00 MiB`, writer `384.71 MiB`
- Exact validation passed with `rust_minus_duckdb=0` and `duckdb_minus_rust=0`.

Dense Utf8 interleaved stress:

`RPM_BENCH_SCENARIO=ordered_payload_string_pragmatic RPM_BENCH_TARGET_INPUT_GIB=1 RPM_BENCH_MEASURED_RUNS=1 RPM_BENCH_RUST_PARALLELISM=0 RPM_BENCH_RUST_COMPRESSION=snappy RPM_BENCH_RUST_ORDERED_MEMORY_BUDGET_MIB=4096 RPM_BENCH_EXACT_VALIDATION=true cargo run --release --example rust_vs_duckdb_benchmark`

- total input: `853.43 MiB`; rows: `30,471,588`
- Rust resolved parallelism: `6`
- Rust `1062 ms`, output `300,726,369` bytes, Snappy
- Rust peak RSS `3593.33 MiB`; CPU `11930 ms` total (`9111 ms` user, `2819 ms` sys, `1122%` of wall)
- DuckDB `1068 ms`, output `225,976,144` bytes, Snappy
- DuckDB peak RSS `4292.73 MiB`; CPU `9748 ms` total (`8766 ms` user, `981 ms` sys, `912%` of wall)
- Delta: Rust `-6 ms` wall, `-699.40 MiB` peak RSS, `+2182 ms` total CPU
- Rust ordered breakdown: merge `563 ms`, decode `1603 ms`, prepare `157 ms`, assembly `2554 ms` (`1019 ms` selection, `1535 ms` materialization work, `3 ms` materialization wait), writer elapsed `1017 ms`, encode work `10258 ms`, sink `317 ms`, close `1 ms`
- Dense path: `117` partition jobs, `30,471,488` rows, `1019 ms` selection work, `1491 ms` materialization work, `0` fallbacks
- Buffer peaks: ordered pipeline `2530.00 MiB`, writer `221.56 MiB`
- Exact validation passed with `rust_minus_duckdb=0` and `duckdb_minus_rust=0`.

Dense LargeUtf8 interleaved stress:

`RPM_BENCH_SCENARIO=ordered_payload_large_string_pragmatic RPM_BENCH_TARGET_INPUT_GIB=1 RPM_BENCH_MEASURED_RUNS=1 RPM_BENCH_RUST_PARALLELISM=0 RPM_BENCH_RUST_COMPRESSION=snappy RPM_BENCH_RUST_ORDERED_MEMORY_BUDGET_MIB=4096 RPM_BENCH_EXACT_VALIDATION=true cargo run --release --example rust_vs_duckdb_benchmark`

- total input: `853.43 MiB`; rows: `30,471,588`
- Rust resolved parallelism: `6`
- Rust `1122 ms`, output `300,726,369` bytes, Snappy
- Rust peak RSS `3729.92 MiB`; CPU `12729 ms` total (`9664 ms` user, `3064 ms` sys, `1135%` of wall)
- DuckDB `965 ms`, output `225,976,144` bytes, Snappy
- DuckDB peak RSS `4156.19 MiB`; CPU `9838 ms` total (`8814 ms` user, `1023 ms` sys, `1019%` of wall)
- Delta: Rust `+157 ms` wall, `-426.27 MiB` peak RSS, `+2891 ms` total CPU
- Rust ordered breakdown: merge `588 ms`, decode `1759 ms`, prepare `153 ms`, assembly `2750 ms` (`1097 ms` selection, `1652 ms` materialization work, `0 ms` materialization wait), writer elapsed `1071 ms`, encode work `10989 ms`, sink `308 ms`, close `1 ms`
- Dense path: `117` partition jobs, `30,471,488` rows, `1097 ms` selection work, `1607 ms` materialization work, `0` fallbacks
- Buffer peaks: ordered pipeline `2933.00 MiB`, writer `266.61 MiB`
- Exact validation passed with `rust_minus_duckdb=0` and `duckdb_minus_rust=0`. DuckDB normalizes `LargeUtf8` to `Utf8` in the benchmark schema shape, so validation treats those string widths as compatible.

Mixed Utf8 ordered target:

`RPM_BENCH_SCENARIO=ordered_payload_string_mixed_pragmatic RPM_BENCH_TARGET_INPUT_GIB=1 RPM_BENCH_MEASURED_RUNS=1 RPM_BENCH_RUST_PARALLELISM=0 RPM_BENCH_RUST_COMPRESSION=snappy RPM_BENCH_RUST_ORDERED_MEMORY_BUDGET_MIB=4096 RPM_BENCH_EXACT_VALIDATION=true cargo run --release --example rust_vs_duckdb_benchmark`

- total input: `628.33 MiB`; rows: `70,609,416`
- Rust resolved parallelism: `6`
- Rust `860 ms`, output `663,229,876` bytes, Snappy
- Rust peak RSS `1155.98 MiB`; CPU `2654 ms` total (`1830 ms` user, `824 ms` sys, `309%` of wall)
- DuckDB `2018 ms`, output `779,807,720` bytes, Snappy
- DuckDB peak RSS `8156.39 MiB`; CPU `19685 ms` total (`17584 ms` user, `2100 ms` sys, `975%` of wall)
- Delta: Rust `-1158 ms` wall, `-7000.41 MiB` peak RSS, `-17031 ms` total CPU
- Copy path: `60` copied row groups, `60,470,532` copied rows, `562,732,276` copied bytes, copy time `418 ms`
- Dense path: `66` partition jobs, `7,060,932` rows, `52 ms` selection work, `275 ms` materialization work, `0` fallbacks
- Buffer peaks: ordered pipeline `1504.00 MiB`, writer `661.86 MiB`
- Exact validation passed with `rust_minus_duckdb=0` and `duckdb_minus_rust=0`.

Important: the comparison should be run in `--release`. A debug-mode `cargo run` makes the Rust merge path look artificially slow and is not a fair comparison against the optimized DuckDB CLI binary.

For a much larger startup-latency-insensitive run, a 10 GiB target is:

`RPM_BENCH_SCENARIO=top_level_pragmatic RPM_BENCH_TARGET_INPUT_GIB=10 RPM_BENCH_MEASURED_RUNS=1 cargo run --release --example rust_vs_duckdb_benchmark`
