use std::fs::File;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;

/// This helper function is the same as in the previous example.
/// It takes a single RecordBatch and a target schema, and adds null arrays
/// for any columns that are in the target schema but not in the batch.
fn adjust_record_batch(
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
    let reader1_builder = ParquetRecordBatchReaderBuilder::try_new(file1_meta)?;
    let schema1 = reader1_builder.schema();

    let file2_meta = File::open("file2.parquet")?;
    let reader2_builder = ParquetRecordBatchReaderBuilder::try_new(file2_meta)?;
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

    // 3. Create the writer for the output file with the merged schema
    let file_out = File::create("merged_streaming.parquet")?;
    let mut writer = ArrowWriter::try_new(file_out, merged_schema.clone(), None)?;

    // Start timing the merge operation
    let merge_start = Instant::now();

    // 4. Stream from file 1, adjust, and write to the output
    let file1_to_stream = File::open("file1.parquet")?;
    let reader1 = ParquetRecordBatchReaderBuilder::try_new(file1_to_stream)?.build()?;

    for batch_result in reader1 {
        let batch = batch_result?;
        let adjusted_batch = adjust_record_batch(batch, merged_schema.clone())?;
        writer.write(&adjusted_batch)?;
    }

    // 5. Stream from file 2, adjust, and write to the output
    let file2_to_stream = File::open("file2.parquet")?;
    let reader2 = ParquetRecordBatchReaderBuilder::try_new(file2_to_stream)?.build()?;

    for batch_result in reader2 {
        let batch = batch_result?;
        let adjusted_batch = adjust_record_batch(batch, merged_schema.clone())?;
        writer.write(&adjusted_batch)?;
    }

    // 6. Finalize the output file
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
