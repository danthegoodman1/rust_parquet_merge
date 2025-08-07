use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- Setup: Create two sample parquet files (same as before) ----
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
    let file1 = File::create("file1.parquet")?;
    let mut writer1 = ArrowWriter::try_new(file1, schema1, Some(WriterProperties::new()))?;
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
    let file2 = File::create("file2.parquet")?;
    let mut writer2 = ArrowWriter::try_new(file2, schema2, Some(WriterProperties::new()))?;
    writer2.write(&batch2)?;
    writer2.close()?;

    // ---- Print contents of input files before merge ----
    println!("\n=== Contents of file1.parquet ===");
    let file1_read = File::open("file1.parquet")?;
    let reader1_print = ParquetRecordBatchReaderBuilder::try_new(file1_read)?.build()?;
    for batch_result in reader1_print {
        let batch = batch_result?;
        println!("{:#?}", batch);
    }

    println!("\n=== Contents of file2.parquet ===");
    let file2_read = File::open("file2.parquet")?;
    let reader2_print = ParquetRecordBatchReaderBuilder::try_new(file2_read)?.build()?;
    for batch_result in reader2_print {
        let batch = batch_result?;
        println!("{:#?}", batch);
    }

    // ---- Streaming Merge Logic ----

    // 1. Get schemas from file metadata without reading data
    let file1_meta = File::open("file1.parquet")?;
    let mut reader1_builder = ParquetRecordBatchReaderBuilder::try_new(file1_meta)?;
    let schema1 = reader1_builder.schema();

    let file2_meta = File::open("file2.parquet")?;
    let mut reader2_builder = ParquetRecordBatchReaderBuilder::try_new(file2_meta)?;
    let schema2 = reader2_builder.schema();

    // 2. Merge schemas to create the target schema for the output file
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

    // 4. Create the writer for the output file with the merged schema (buffered I/O)
    let file_out = File::create("merged_streaming.parquet")?;
    let buf_out = BufWriter::with_capacity(1 << 20, file_out);
    let mut writer = ArrowWriter::try_new(buf_out, merged_schema.clone(), None)?;

    // Start timing the merge operation
    let merge_start = Instant::now();

    // 5. Optional: set a larger batch size to reduce per-batch overhead
    // Comment out or tune if needed.
    reader1_builder = reader1_builder.with_batch_size(64 * 1024);
    reader2_builder = reader2_builder.with_batch_size(64 * 1024);

    // 6. Stream from file 1, adjust, and write to the output (reuse existing builder)
    let reader1 = reader1_builder.build()?;

    for batch_result in reader1 {
        let batch = batch_result?;
        let adjusted_batch = adjust_with_mapping(&batch, &merged_schema, &mapping1)?;
        writer.write(&adjusted_batch)?;
    }

    // 7. Stream from file 2, adjust, and write to the output (reuse existing builder)
    let reader2 = reader2_builder.build()?;

    for batch_result in reader2 {
        let batch = batch_result?;
        let adjusted_batch = adjust_with_mapping(&batch, &merged_schema, &mapping2)?;
        writer.write(&adjusted_batch)?;
    }

    // 8. Finalize the output file
    writer.close()?;

    // End timing and report
    let merge_duration = merge_start.elapsed();
    println!("\nParquet files merged successfully via streaming into merged_streaming.parquet");
    println!("⏱️  Merge operation took: {:?}", merge_duration);

    // ---- Print contents of merged file ----
    println!("\n=== Contents of merged_streaming.parquet ===");
    let merged_file = File::open("merged_streaming.parquet")?;
    let merged_reader = ParquetRecordBatchReaderBuilder::try_new(merged_file)?.build()?;
    for batch_result in merged_reader {
        let batch = batch_result?;
        println!("{:#?}", batch);
    }

    Ok(())
}
