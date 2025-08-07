# Parquet streaming merge example

Fastest of 3 runs (with `--release`, M3 Max MBP):

Parquet + Arrow crates: `246-616µs`

Parquet + Arrow crates (async): `550-900us`

Polars: `8.977625ms` 🫣 _I hope this isn't the most efficient way?_

Aisle: `657.458µs-1.6ms` (but jumped up to 1ms, more variance)
