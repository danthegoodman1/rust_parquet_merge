# Parquet streaming merge example

Fastest of 3 runs (with `--release`, M3 Max MBP):

Parquet + Arrow crates: `246-616µs`

Parquet + Arrow crates (async): `550-900us`

Polars: `8.977625ms` 🫣 _I hope this isn't the most efficient way?_

Aisle: `463-820us` (but jumped up to 1ms, more variance)


Aisle and parquet async are effectively the same, any difference for this use case is up to error bounds, async scheduling, etc.

Note: with recent rust updates these are much faster now, roughly twice as fast for the parquet + arrow crates on the same machine
