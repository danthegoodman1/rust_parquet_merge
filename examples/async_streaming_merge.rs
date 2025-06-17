use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures_util::StreamExt;
use parquet::arrow::ArrowWriter;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_writer::AsyncArrowWriter;
use parquet::file::properties::WriterProperties;
use tokio::fs::File;
use tokio::io::AsyncWrite;

/// Async helper function to adjust a RecordBatch to match a target schema.
/// This adds null arrays for any columns that are in the target schema but not in the batch.
async fn adjust_record_batch(
    batch: RecordBatch,
    target_schema: SchemaRef,
) -> Result<RecordBatch, parquet::errors::ParquetError> {
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

    RecordBatch::try_new(target_schema, new_columns)
        .map_err(|e| parquet::errors::ParquetError::General(format!("Arrow error: {}", e)))
}

/// Async function to create a sample parquet file
async fn create_sample_file(
    filename: &str,
    schema: SchemaRef,
    batch: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    // For creating files, we still need to use sync I/O since ArrowWriter requires std::io::Write
    let file = std::fs::File::create(filename)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::new()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Async function to print the contents of a parquet file
async fn print_parquet_contents(
    filename: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== {} ===", title);

    let file = File::open(filename).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let mut stream = builder.build()?;

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        println!("{:#?}", batch);
    }

    Ok(())
}

/// Async function to get schema from a parquet file
async fn get_parquet_schema(filename: &str) -> Result<SchemaRef, Box<dyn std::error::Error>> {
    let file = File::open(filename).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    Ok(builder.schema().clone())
}

/// Async function to stream and process batches from a parquet file
async fn stream_and_write_batches<W>(
    filename: &str,
    writer: &mut AsyncArrowWriter<W>,
    target_schema: SchemaRef,
) -> Result<(), Box<dyn std::error::Error>>
where
    W: AsyncWrite + Unpin + Send,
{
    let file = File::open(filename).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let mut stream = builder.build()?;

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        let adjusted_batch = adjust_record_batch(batch, target_schema.clone()).await?;
        writer.write(&adjusted_batch).await?;
    }

    Ok(())
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

    // Create sample files asynchronously
    tokio::try_join!(
        create_sample_file("async_file1.parquet", schema1.clone(), batch1),
        create_sample_file("async_file2.parquet", schema2.clone(), batch2)
    )?;

    // ---- Print contents of input files before merge ----
    tokio::try_join!(
        print_parquet_contents("async_file1.parquet", "Contents of async_file1.parquet"),
        print_parquet_contents("async_file2.parquet", "Contents of async_file2.parquet")
    )?;

    // ---- Async Streaming Merge Logic ----

    // 1. Get schemas from both files concurrently
    let (schema1, schema2) = tokio::try_join!(
        get_parquet_schema("async_file1.parquet"),
        get_parquet_schema("async_file2.parquet")
    )?;

    // 2. Merge schemas to create the target schema for the output file
    let mut merged_fields: Vec<_> = schema1.fields().iter().cloned().collect();
    for field2 in schema2.fields() {
        if schema1.field_with_name(field2.name()).is_err() {
            merged_fields.push(field2.clone());
        }
    }
    let merged_schema = Arc::new(Schema::new(merged_fields));
    println!("Merged schema created successfully.");

    // 3. Create the async writer for the output file with the merged schema
    let output_file = File::create("async_merged_streaming.parquet").await?;
    let mut writer = AsyncArrowWriter::try_new(output_file, merged_schema.clone(), None)?;

    // Start timing the merge operation
    let merge_start = Instant::now();

    // 4. Stream from both files and write to the output
    // Process file1 first
    stream_and_write_batches("async_file1.parquet", &mut writer, merged_schema.clone()).await?;

    // Then process file2
    stream_and_write_batches("async_file2.parquet", &mut writer, merged_schema.clone()).await?;

    // 5. Finalize the output file
    writer.close().await?;

    // End timing and report
    let merge_duration = merge_start.elapsed();
    println!(
        "\nParquet files merged successfully via async streaming into async_merged_streaming.parquet"
    );

    // ---- Print contents of merged file ----
    print_parquet_contents(
        "async_merged_streaming.parquet",
        "Contents of async_merged_streaming.parquet",
    )
    .await?;

    println!("⏱️  Merge operation took: {:?}", merge_duration);
    Ok(())
}
