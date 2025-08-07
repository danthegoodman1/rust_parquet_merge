use std::io::BufWriter;
use std::sync::Arc;
use std::time::Instant;

use aisle::{ArrowReaderOptions, ParquetRecordBatchStreamBuilder};
use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures_util::StreamExt;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tokio::fs::File as TokioFile;

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

/// Open a parquet file and return an Aisle builder initialized with options.
async fn open_aisle_builder(
    file_path: &str,
) -> Result<ParquetRecordBatchStreamBuilder<TokioFile>, Box<dyn std::error::Error>> {
    let file = TokioFile::open(file_path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        file,
        ArrowReaderOptions::new().with_page_index(true),
    )
    .await?;
    Ok(builder)
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
    let builder_demo = open_aisle_builder("aisle_file1.parquet").await?;
    let mut stream = builder_demo.build()?;
    let mut total_rows = 0;
    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        total_rows += batch.num_rows();
        println!("Batch with {} rows: {:?}", batch.num_rows(), batch);
    }
    println!("⏱️  Read {} rows in {:?}", total_rows, start_time.elapsed());

    // ---- Schema merging and streaming merge ----
    println!("\n=== Performing Aisle Streaming Merge ===");

    // Open builders once and get schemas
    let (builder1, builder2) = tokio::try_join!(
        open_aisle_builder("aisle_file1.parquet"),
        open_aisle_builder("aisle_file2.parquet"),
    )?;
    let schema1_aisle = builder1.schema().clone();
    let schema2_aisle = builder2.schema().clone();

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

    // Precompute index mappings
    let mapping1 = build_index_mapping(schema1_aisle.as_ref(), merged_schema.as_ref());
    let mapping2 = build_index_mapping(schema2_aisle.as_ref(), merged_schema.as_ref());

    // Create output writer (buffered)
    let file_out = std::fs::File::create("aisle_merged.parquet")?;
    let buf_out = BufWriter::with_capacity(1 << 20, file_out);
    let mut writer = ArrowWriter::try_new(buf_out, merged_schema.clone(), None)?;

    let merge_start = Instant::now();

    // Stream from file1 and write to output
    println!("Streaming data from file1...");
    let mut stream1 = builder1.build()?;
    while let Some(batch_result) = stream1.next().await {
        let batch = batch_result?;
        let adjusted_batch = adjust_with_mapping(&batch, &merged_schema, &mapping1)?;
        writer.write(&adjusted_batch)?;
    }

    // Stream from file2 and write to output
    println!("Streaming data from file2...");
    let mut stream2 = builder2.build()?;
    while let Some(batch_result) = stream2.next().await {
        let batch = batch_result?;
        let adjusted_batch = adjust_with_mapping(&batch, &merged_schema, &mapping2)?;
        writer.write(&adjusted_batch)?;
    }

    writer.close()?;
    let merge_duration = merge_start.elapsed();

    println!("\n✅ Parquet files merged successfully with Aisle into aisle_merged.parquet");
    println!("⏱️  Merge operation took: {:?}", merge_duration);

    // ---- Show final merged results ----
    println!("\n=== Contents of merged file (via Aisle) ===");
    let final_builder = open_aisle_builder("aisle_merged.parquet").await?;
    let mut final_stream = final_builder.build()?;
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
