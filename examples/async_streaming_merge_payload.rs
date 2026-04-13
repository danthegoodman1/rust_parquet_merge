use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use parquet::arrow::async_writer::AsyncArrowWriter;
use parquet::file::properties::WriterProperties;
use rust_parquet_merge::{PayloadMergeOptions, merge_payload_parquet_files};
use tokio::fs::File;

fn payload_sample_inputs() -> (SchemaRef, RecordBatch, SchemaRef, RecordBatch) {
    let left_profile_fields: Fields =
        vec![Arc::new(Field::new("name", DataType::Utf8, true))].into();
    let left_profile_array = Arc::new(StructArray::new(
        left_profile_fields.clone(),
        vec![Arc::new(StringArray::from(vec![
            Some("Alice"),
            Some("Bob"),
            Some("Charlie"),
        ])) as ArrayRef],
        None,
    )) as ArrayRef;
    let left_payload_fields: Fields = vec![
        Arc::new(Field::new("score", DataType::Int32, true)),
        Arc::new(Field::new(
            "profile",
            DataType::Struct(left_profile_fields.clone()),
            true,
        )),
    ]
    .into();
    let left_payload_array = Arc::new(StructArray::new(
        left_payload_fields.clone(),
        vec![
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30)])) as ArrayRef,
            left_profile_array,
        ],
        None,
    )) as ArrayRef;
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int32, false),
        Field::new("org_id", DataType::Int32, false),
        Field::new("payload", DataType::Struct(left_payload_fields), true),
    ]));
    let left_batch = RecordBatch::try_new(
        left_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(Int32Array::from(vec![10, 10, 20])) as ArrayRef,
            left_payload_array,
        ],
    )
    .unwrap();

    let right_profile_fields: Fields =
        vec![Arc::new(Field::new("tier", DataType::Utf8, true))].into();
    let right_profile_array = Arc::new(StructArray::new(
        right_profile_fields.clone(),
        vec![Arc::new(StringArray::from(vec![Some("gold"), None])) as ArrayRef],
        None,
    )) as ArrayRef;
    let right_payload_fields: Fields = vec![
        Arc::new(Field::new("score", DataType::Float64, true)),
        Arc::new(Field::new(
            "profile",
            DataType::Struct(right_profile_fields.clone()),
            true,
        )),
    ]
    .into();
    let right_payload_array = Arc::new(StructArray::new(
        right_payload_fields.clone(),
        vec![
            Arc::new(Float64Array::from(vec![Some(44.5), Some(8.25)])) as ArrayRef,
            right_profile_array,
        ],
        None,
    )) as ArrayRef;
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("org_id", DataType::Int32, false),
        Field::new("event_id", DataType::Int32, false),
        Field::new("payload", DataType::Struct(right_payload_fields), true),
    ]));
    let right_batch = RecordBatch::try_new(
        right_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![30, 40])) as ArrayRef,
            Arc::new(Int32Array::from(vec![4, 5])) as ArrayRef,
            right_payload_array,
        ],
    )
    .unwrap();

    (left_schema, left_batch, right_schema, right_batch)
}

async fn create_sample_file(
    filename: &str,
    schema: SchemaRef,
    batch: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(filename).await?;
    let mut writer = AsyncArrowWriter::try_new(file, schema, Some(WriterProperties::new()))?;
    writer.write(&batch).await?;
    writer.close().await?;
    Ok(())
}

async fn read_parquet_batches(
    filename: &str,
) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
    let file = File::open(filename).await?;
    let builder = parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder::new(file).await?;
    let mut stream = builder.build()?;
    let mut batches = Vec::new();

    while let Some(batch_result) = futures_util::StreamExt::next(&mut stream).await {
        batches.push(batch_result?);
    }

    Ok(batches)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (left_schema, left_batch, right_schema, right_batch) = payload_sample_inputs();
    let options = PayloadMergeOptions::default();

    tokio::try_join!(
        create_sample_file("async_payload_file1.parquet", left_schema, left_batch),
        create_sample_file("async_payload_file2.parquet", right_schema, right_batch)
    )?;

    let report = merge_payload_parquet_files(
        &[
            "async_payload_file1.parquet".into(),
            "async_payload_file2.parquet".into(),
        ],
        std::path::Path::new("async_merged_streaming_payload.parquet"),
        &options,
    )
    .await?;
    println!(
        "Payload merge wrote async_merged_streaming_payload.parquet in {:?} \
         (planning {:?}, rows {})",
        report.total_duration, report.planning_duration, report.rows,
    );

    let merged_batches = read_parquet_batches("async_merged_streaming_payload.parquet").await?;
    let merged_schema = merged_batches[0].schema();
    let ctx = SessionContext::new();
    let mem_table = MemTable::try_new(merged_schema, vec![merged_batches])?;
    ctx.register_table("merged_payload", Arc::new(mem_table))?;

    let aggregate_batches = ctx
        .sql("SELECT sum(payload['score']) AS total_score FROM merged_payload")
        .await?
        .collect()
        .await?;
    println!("\n=== DataFusion aggregate over payload['score'] ===");
    arrow::util::pretty::print_batches(&aggregate_batches)?;

    let projection_batches = ctx
        .sql(
            "SELECT event_id, payload['profile']['name'] AS profile_name \
             FROM merged_payload ORDER BY event_id",
        )
        .await?
        .collect()
        .await?;
    println!("\n=== DataFusion projection over payload['profile']['name'] ===");
    arrow::util::pretty::print_batches(&projection_batches)?;

    Ok(())
}
