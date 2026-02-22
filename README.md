# Parquet streaming merge example

Fastest of 3 runs (with `--release`, M3 Max MBP):

Parquet + Arrow crates: `246-616µs`

Parquet + Arrow crates (async): `550-900us`

Polars: `8.977625ms` 🫣 _I hope this isn't the most efficient way?_

Aisle: `463-820us` (but jumped up to 1ms, more variance)


Aisle and parquet async are effectively the same, any difference for this use case is up to error bounds, async scheduling, etc.

Note: with recent rust updates these are much faster now, roughly twice as fast for the parquet + arrow crates on the same machine

## Async widening merge example

There is now a widening-focused async merge example at `examples/async_streaming_merge_widening.rs`.

Run it with:

`cargo run --example async_streaming_merge_widening`

Run the widening tests with:

`cargo test --example async_streaming_merge_widening`

### Widening behavior

- Primitive widening is permissive: numeric types are promoted to a common type and primitive/string combinations can promote to string.
- Structured widening is shape-aware: `Struct + Struct` unions fields by name, widens shared children recursively, and makes missing fields nullable.
- List widening is supported for `List` and `LargeList`, including mixed combinations where the merged type is promoted to `LargeList` when needed.
- Primitive vs structured conflicts remain strict by default (for example `Utf8` vs `Struct` returns an incompatibility error).
