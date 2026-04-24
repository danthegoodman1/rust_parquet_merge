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
- `RPM_BENCH_RUST_SIZE_PROFILE=none`, `snappy_column_tuned`, or `zstd1_column_tuned` to apply benchmark-only per-column size tuning
- `RPM_BENCH_RUST_COLUMN_DICTIONARY=path.to.leaf=false,path.to.other=true` to override dictionary encoding per Parquet leaf column
- `RPM_BENCH_RUST_COLUMN_ENCODING=path.to.leaf=plain|delta_binary_packed|delta_length_byte_array|delta_byte_array|byte_stream_split`
- `RPM_BENCH_RUST_COLUMN_COMPRESSION=path.to.leaf=snappy|uncompressed|lz4_raw|zstd:1`
- `RPM_BENCH_RUST_WRITER_VERSION=parquet1` or `parquet2`
- `RPM_BENCH_RUST_DATA_PAGE_SIZE_LIMIT=<int>`, `RPM_BENCH_RUST_DICTIONARY_PAGE_SIZE_LIMIT=<int>`, and `RPM_BENCH_RUST_DATA_PAGE_ROW_COUNT_LIMIT=<int>` to tune Parquet page sizing
- `RPM_BENCH_RUST_READ_BATCH_SIZE=<int>` to tune Rust Parquet read batch size
- `RPM_BENCH_RUST_OUTPUT_BATCH_ROWS=<int>` to tune Rust output batch materialization size
- `RPM_BENCH_RUST_OUTPUT_ROW_GROUP_ROWS=<int>` to tune Rust output row group size
- `RPM_BENCH_RUST_PREFETCH_BATCHES_PER_SOURCE=<int>` to tune per-source buffering
- `RPM_BENCH_RUST_ORDERED_MEMORY_BUDGET_MIB=<int>` to opt ordered merge into the higher-throughput byte-bounded pipeline; `4096` is the current speed target
- `RPM_BENCH_EXACT_VALIDATION=true` to add DuckDB `EXCEPT ALL` validation both ways
- `RPM_BENCH_DUCKDB_THREADS=<int>` to set DuckDB `PRAGMA threads`
- `RPM_BENCH_DUCKDB_COMPRESSION=snappy`, `uncompressed`, `zstd`, or `lz4` to set DuckDB output compression

### Current benchmark snapshot

Environment: M3 Max MBP, DuckDB CLI `v1.4.1`, `--release`, one measured run. Ordered runs use `RPM_BENCH_RUST_PARALLELISM=0`, `RPM_BENCH_RUST_ORDERED_MEMORY_BUDGET_MIB=4096`, and `RPM_BENCH_EXACT_VALIDATION=true`.

The accepted size profile is `RPM_BENCH_RUST_SIZE_PROFILE=snappy_column_tuned`. It keeps DuckDB-comparable Snappy output, disables dictionaries on high-cardinality ordered/string/list leaves, and leaves mixed row-group-copy scenarios copy-compatible. Exact validation passed both directions for every ordered row below.

Top-level 0.25 GiB smoke:

| Rust profile | Rust median | Rust output bytes | Rust peak RSS | Rust CPU | DuckDB median | DuckDB output bytes | DuckDB peak RSS | DuckDB CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `snappy_column_tuned` | `241 ms` | `127,195,626` | `555.39 MiB` | `1236 ms` (`512%`) | `371 ms` | `115,002,049` | `1109.73 MiB` | `1748 ms` (`471%`) |
| `zstd1_column_tuned` | `258 ms` | `64,307,192` | `550.77 MiB` | `1414 ms` (`547%`) | `351 ms` | `115,002,049` | `1115.83 MiB` | `1919 ms` (`545%`) |

Ordered 1 GiB snapshots:

| Scenario | Rust profile | Rust median | Rust output bytes | Rust peak RSS | Rust CPU | DuckDB median | DuckDB output bytes | DuckDB peak RSS | DuckDB CPU | Copy path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mixed Int64 target | `snappy_column_tuned` | `826 ms` | `651,237,782` | `965.89 MiB` | `2716 ms` (`329%`) | `1378 ms` | `755,614,770` | `6593.30 MiB` | `12268 ms` (`890%`) | `60` row groups, `62,292,672` rows |
| Dense Int64 stress | `snappy_column_tuned` | `787 ms` | `296,574,787` | `3549.36 MiB` | `9041 ms` (`1148%`) | `1189 ms` | `315,281,804` | `5749.86 MiB` | `10386 ms` (`873%`) | none |
| Dense Utf8 stress | `snappy_column_tuned` | `808 ms` | `232,577,720` | `3956.91 MiB` | `8221 ms` (`1017%`) | `1072 ms` | `225,976,144` | `4625.48 MiB` | `9908 ms` (`924%`) | none |
| Dense LargeUtf8 stress | `snappy_column_tuned` | `822 ms` | `232,577,720` | `3705.91 MiB` | `8224 ms` (`1000%`) | `1097 ms` | `225,976,144` | `4464.92 MiB` | `9976 ms` (`909%`) | none |
| Mixed Utf8 target | `snappy_column_tuned` | `987 ms` | `663,454,541` | `1415.00 MiB` | `2847 ms` (`288%`) | `2134 ms` | `779,807,779` | `8029.09 MiB` | `20876 ms` (`978%`) | `60` row groups, `60,470,532` rows |

`zstd1_column_tuned` is excellent for output size but is not the default accepted ordered string profile: on the 1 GiB dense Utf8 run it produced `39,467,842` bytes but took `1341 ms`, and on dense LargeUtf8 it took `2986 ms`. Use it when size matters more than dense string wall time.

The benchmark JSON includes per-column byte deltas plus cumulative row-group encode worker time and ordered materialization work. Worker-time values are summed across parallel workers, so they can be larger than elapsed wall time.

Important: the comparison should be run in `--release`. A debug-mode `cargo run` makes the Rust merge path look artificially slow and is not a fair comparison against the optimized DuckDB CLI binary.

For a much larger startup-latency-insensitive run, a 10 GiB target is:

`RPM_BENCH_SCENARIO=top_level_pragmatic RPM_BENCH_TARGET_INPUT_GIB=10 RPM_BENCH_MEASURED_RUNS=1 cargo run --release --example rust_vs_duckdb_benchmark`
