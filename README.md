# Parquet streaming merge example

Fastest of 3 runs (with `--release`, M3 Max MBP):

Parquet + Arrow crates: `333.084µs`

Polars: `8.977625ms` 🫣 _I hope this isn't the most efficient way?_

Aisle: `657.458µs` (but jumped up to 1ms, more variance)
