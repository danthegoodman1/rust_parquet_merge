use polars::prelude::*;
use std::time::Instant;

fn main() -> PolarsResult<()> {
    // Create sample data for file1.parquet (id, name)
    let df1 = df! {
        "id" => [1, 2, 3],
        "name" => [Some("Alice"), Some("Bob"), Some("Charlie")],
    }?;

    // Write file1.parquet
    let mut file1 = std::fs::File::create("file1.parquet")?;
    ParquetWriter::new(&mut file1).finish(&mut df1.clone())?;
    println!("Created file1.parquet");

    // Create sample data for file2.parquet (id, age) - disjoint columns
    let df2 = df! {
        "id" => [4, 5],
        "age" => [Some(30), None::<i32>],
    }?;

    // Write file2.parquet
    let mut file2 = std::fs::File::create("file2.parquet")?;
    ParquetWriter::new(&mut file2).finish(&mut df2.clone())?;
    println!("Created file2.parquet");

    // Print contents of input files before merge
    println!("\n=== Contents of file1.parquet ===");
    println!("{}", df1);

    println!("\n=== Contents of file2.parquet ===");
    println!("{}", df2);

    // Scan the files lazily (doesn't load data into memory)
    let lf1 = LazyFrame::scan_parquet("file1.parquet", Default::default())?;
    let lf2 = LazyFrame::scan_parquet("file2.parquet", Default::default())?;

    // Start timing the merge operation
    let merge_start = Instant::now();

    // Diagonally concatenate the lazy frames. This handles disjoint columns.
    let merged_lf = concat(
        &[lf1, lf2],
        UnionArgs {
            rechunk: false,
            parallel: true,
            to_supertypes: false,
            diagonal: true, // This is the key for disjoint columns
            from_partitioned_ds: false,
            maintain_order: false,
        },
    )?;

    // Create a new parquet file by collecting the result of the lazy computation
    let mut file = std::fs::File::create("merged_polars.parquet")?;
    ParquetWriter::new(&mut file).finish(&mut merged_lf.collect()?)?;

    // End timing and report
    let merge_duration = merge_start.elapsed();
    println!("\nParquet files merged successfully with Polars into merged_polars.parquet");
    println!("⏱️  Merge operation took: {:?}", merge_duration);

    // Print contents of merged file
    println!("\n=== Contents of merged_polars.parquet ===");
    let merged_df =
        LazyFrame::scan_parquet("merged_polars.parquet", Default::default())?.collect()?;
    println!("{}", merged_df);

    Ok(())
}
