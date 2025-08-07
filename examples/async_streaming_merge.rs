use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures_util::StreamExt;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_writer::AsyncArrowWriter;
use parquet::file::properties::WriterProperties;
use tokio::fs::File;
use tokio::io::{AsyncWrite, BufWriter};

/// Build a one-time column index mapping from a source schema to a target schema.
/// Each entry is `Some(source_index)` if the target field exists in the source, otherwise `None`.
fn build_index_mapping(source_schema: &Schema, target_schema: &Schema) -> Vec<Option<usize>> {
    target_schema
        .fields()
        .iter()
        .map(|target_field| source_schema.index_of(target_field.name()).ok())
        .collect()
}

/// Adjust a `RecordBatch` to match `target_schema` using a precomputed index mapping.
fn adjust_with_mapping(
    batch: &RecordBatch,
    target_schema: &SchemaRef,
    mapping: &[Option<usize>],
) -> Result<RecordBatch, parquet::errors::ParquetError> {
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(mapping.len());

    for (i, maybe_src_idx) in mapping.iter().enumerate() {
        match maybe_src_idx {
            Some(src_idx) => new_columns.push(batch.column(*src_idx).clone()),
            None => new_columns.push(arrow::array::new_null_array(
                target_schema.field(i).data_type(),
                batch.num_rows(),
            )),
        }
    }

    RecordBatch::try_new(target_schema.clone(), new_columns)
        .map_err(|e| parquet::errors::ParquetError::General(format!("Arrow error: {}", e)))
}

/// Async function to create a sample parquet file
async fn create_sample_file(
    filename: &str,
    schema: SchemaRef,
    batch: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    // Using async I/O for consistency
    let file = File::create(filename).await?;
    let mut writer = AsyncArrowWriter::try_new(file, schema, Some(WriterProperties::new()))?;
    writer.write(&batch).await?;
    writer.close().await?;
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

/// Helper to open a parquet file and return its async builder.
async fn open_parquet_builder(
    filename: &str,
) -> Result<ParquetRecordBatchStreamBuilder<File>, Box<dyn std::error::Error>> {
    let file = File::open(filename).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    Ok(builder)
}

/// Stream and write all batches from a builder using a precomputed mapping.
async fn stream_and_write_from_builder<W>(
    builder: ParquetRecordBatchStreamBuilder<File>,
    writer: &mut AsyncArrowWriter<W>,
    target_schema: &SchemaRef,
    mapping: &[Option<usize>],
) -> Result<(), Box<dyn std::error::Error>>
where
    W: AsyncWrite + Unpin + Send,
{
    let mut stream = builder.build()?;

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        let adjusted_batch = adjust_with_mapping(&batch, target_schema, mapping)?;
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

    // Create sample files asynchronously with proper async I/O
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

    // 1. Open both files and get builders concurrently (avoid double opens later)
    let (builder1, builder2) = tokio::try_join!(
        open_parquet_builder("async_file1.parquet"),
        open_parquet_builder("async_file2.parquet")
    )?;

    // 2. Merge schemas to create the target schema for the output file
    let schema1 = builder1.schema().clone();
    let schema2 = builder2.schema().clone();
    let mut merged_fields: Vec<_> = schema1.fields().iter().cloned().collect();
    for field2 in schema2.fields() {
        if schema1.field_with_name(field2.name()).is_err() {
            merged_fields.push(field2.clone());
        }
    }
    let merged_schema = Arc::new(Schema::new(merged_fields));
    println!("Merged schema created successfully.");

    // 3. Precompute column index mappings (source -> target) for both files
    let mapping1 = build_index_mapping(schema1.as_ref(), merged_schema.as_ref());
    let mapping2 = build_index_mapping(schema2.as_ref(), merged_schema.as_ref());

    // 4. Create the async writer for the output file with buffered I/O
    let output_file = File::create("async_merged_streaming.parquet").await?;
    let output_file = BufWriter::with_capacity(1 << 20, output_file);
    let mut writer = AsyncArrowWriter::try_new(output_file, merged_schema.clone(), None)?;

    // Start timing the merge operation
    let merge_start = Instant::now();

    // 5. Stream from both files using the existing builders and write to the output
    // Process file1 first
    stream_and_write_from_builder(builder1, &mut writer, &merged_schema, &mapping1).await?;

    // Then process file2
    stream_and_write_from_builder(builder2, &mut writer, &merged_schema, &mapping2).await?;

    // 6. Finalize the output file
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
