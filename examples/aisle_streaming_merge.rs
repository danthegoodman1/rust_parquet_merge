use std::sync::Arc;
use std::time::Instant;

use aisle::{ArrowReaderOptions, ParquetRecordBatchStreamBuilder};
use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures_util::StreamExt;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tokio::fs::File as TokioFile;

/// This helper function adjusts a RecordBatch to match a target schema,
/// adding null arrays for missing columns.
fn adjust_record_batch(
    batch: RecordBatch,
    target_schema: SchemaRef,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(target_schema.fields().len());
    let batch_schema = batch.schema();

    for target_field in target_schema.fields() {
        match batch_schema.index_of(target_field.name()) {
            // Column exists in the batch, so we clone it.
            Ok(idx) => {
                new_columns.push(batch.column(idx).clone());
            }
            // Column is missing, so we create a null array for it.
            Err(_) => {
                let null_array =
                    arrow::array::new_null_array(target_field.data_type(), batch.num_rows());
                new_columns.push(null_array);
            }
        }
    }

    Ok(RecordBatch::try_new(target_schema, new_columns)?)
}

/// Create an async Parquet reader using Aisle with tokio::fs::File
async fn create_aisle_reader(
    file_path: &str,
) -> Result<
    impl futures_util::Stream<Item = Result<RecordBatch, parquet::errors::ParquetError>>,
    Box<dyn std::error::Error>,
> {
    let file = TokioFile::open(file_path).await?;

    // Initialize builder with page index enabled for better performance
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        file,
        ArrowReaderOptions::new().with_page_index(true),
    )
    .await?;

    Ok(builder.build()?)
}

/// Get schema from a parquet file using Aisle
async fn get_schema_from_file(file_path: &str) -> Result<SchemaRef, Box<dyn std::error::Error>> {
    let file = TokioFile::open(file_path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        file,
        ArrowReaderOptions::new().with_page_index(true),
    )
    .await?;
    Ok(builder.schema().clone())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- Setup: Create two sample parquet files ----
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![
                Some("Alice"),
                Some("Bob"),
                Some("Charlie"),
            ])),
        ],
    )?;
    let file1 = std::fs::File::create("aisle_file1.parquet")?;
    let mut writer1 = ArrowWriter::try_new(file1, schema1.clone(), Some(WriterProperties::new()))?;
    writer1.write(&batch1)?;
    writer1.close()?;

    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("age", DataType::Int32, true),
    ]));
    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5])),
            Arc::new(Int32Array::from(vec![Some(30), None])),
        ],
    )?;
    let file2 = std::fs::File::create("aisle_file2.parquet")?;
    let mut writer2 = ArrowWriter::try_new(file2, schema2.clone(), Some(WriterProperties::new()))?;
    writer2.write(&batch2)?;
    writer2.close()?;

    // ---- Demonstrate Aisle's async streaming ----
    println!("\n=== Reading with Aisle (async streaming) ===");
    let start_time = Instant::now();
    let mut stream = create_aisle_reader("aisle_file1.parquet").await?;
    let mut total_rows = 0;
    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        total_rows += batch.num_rows();
        println!("Batch with {} rows: {:?}", batch.num_rows(), batch);
    }
    println!("⏱️  Read {} rows in {:?}", total_rows, start_time.elapsed());

    // ---- Schema merging and streaming merge ----
    println!("\n=== Performing Aisle Streaming Merge ===");

    // Get schemas from files
    let schema1_aisle = get_schema_from_file("aisle_file1.parquet").await?;
    let schema2_aisle = get_schema_from_file("aisle_file2.parquet").await?;

    // Merge schemas
    let mut merged_fields: Vec<_> = schema1_aisle.fields().iter().cloned().collect();
    for field2 in schema2_aisle.fields() {
        if schema1_aisle.field_with_name(field2.name()).is_err() {
            merged_fields.push(field2.clone());
        }
    }
    let merged_schema = Arc::new(Schema::new(merged_fields));
    println!(
        "Merged schema created successfully with {} fields",
        merged_schema.fields().len()
    );

    // Create output writer
    let file_out = std::fs::File::create("aisle_merged.parquet")?;
    let mut writer = ArrowWriter::try_new(file_out, merged_schema.clone(), None)?;

    let merge_start = Instant::now();

    // Stream from file1 and write to output
    println!("Streaming data from file1...");
    let mut stream1 = create_aisle_reader("aisle_file1.parquet").await?;
    while let Some(batch_result) = stream1.next().await {
        let batch = batch_result?;
        let adjusted_batch = adjust_record_batch(batch, merged_schema.clone())?;
        writer.write(&adjusted_batch)?;
    }

    // Stream from file2 and write to output
    println!("Streaming data from file2...");
    let mut stream2 = create_aisle_reader("aisle_file2.parquet").await?;
    while let Some(batch_result) = stream2.next().await {
        let batch = batch_result?;
        let adjusted_batch = adjust_record_batch(batch, merged_schema.clone())?;
        writer.write(&adjusted_batch)?;
    }

    writer.close()?;
    let merge_duration = merge_start.elapsed();

    println!("\n✅ Parquet files merged successfully with Aisle into aisle_merged.parquet");
    println!("⏱️  Merge operation took: {:?}", merge_duration);

    // ---- Show final merged results ----
    println!("\n=== Contents of merged file (via Aisle) ===");
    let mut final_stream = create_aisle_reader("aisle_merged.parquet").await?;
    while let Some(batch_result) = final_stream.next().await {
        let batch = batch_result?;
        println!("{:#?}", batch);
    }

    // Clean up temporary files
    let _ = std::fs::remove_file("aisle_file1.parquet");
    let _ = std::fs::remove_file("aisle_file2.parquet");
    let _ = std::fs::remove_file("aisle_merged.parquet");

    Ok(())
}
