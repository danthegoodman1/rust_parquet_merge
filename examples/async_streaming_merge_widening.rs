use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures_util::StreamExt;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_writer::AsyncArrowWriter;
use parquet::file::properties::WriterProperties;
use rust_parquet_merge::{
    WideningOptions, adjust_with_mapping, build_index_mapping, merge_schemas_with_widening,
};
use tokio::fs::File;
use tokio::io::{AsyncWrite, BufWriter};

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

async fn print_parquet_contents(
    filename: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== {title} ===");
    let file = File::open(filename).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let mut stream = builder.build()?;

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        println!("{batch:#?}");
    }

    Ok(())
}

async fn open_parquet_builder(
    filename: &str,
) -> Result<ParquetRecordBatchStreamBuilder<File>, Box<dyn std::error::Error>> {
    let file = File::open(filename).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    Ok(builder)
}

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
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("score", DataType::Int32, true),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30)])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                Some("Alice"),
                Some("Bob"),
                Some("Charlie"),
            ])) as ArrayRef,
        ],
    )?;

    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("score", DataType::Float64, true),
        Field::new("age", DataType::Int32, true),
    ]));
    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5])) as ArrayRef,
            Arc::new(Float64Array::from(vec![Some(44.5), Some(8.25)])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(30), None])) as ArrayRef,
        ],
    )?;

    tokio::try_join!(
        create_sample_file("async_widen_file1.parquet", schema1.clone(), batch1),
        create_sample_file("async_widen_file2.parquet", schema2.clone(), batch2)
    )?;

    tokio::try_join!(
        print_parquet_contents(
            "async_widen_file1.parquet",
            "Contents of async_widen_file1.parquet"
        ),
        print_parquet_contents(
            "async_widen_file2.parquet",
            "Contents of async_widen_file2.parquet"
        )
    )?;

    let (builder1, builder2) = tokio::try_join!(
        open_parquet_builder("async_widen_file1.parquet"),
        open_parquet_builder("async_widen_file2.parquet")
    )?;

    let schema1 = builder1.schema().clone();
    let schema2 = builder2.schema().clone();
    let options = WideningOptions::default();
    let merged_schema = Arc::new(merge_schemas_with_widening(
        schema1.as_ref(),
        schema2.as_ref(),
        &options,
    )?);
    println!("Merged schema with widening created successfully.");

    let mapping1 = build_index_mapping(schema1.as_ref(), merged_schema.as_ref());
    let mapping2 = build_index_mapping(schema2.as_ref(), merged_schema.as_ref());

    let output_file = File::create("async_merged_streaming_widened.parquet").await?;
    let output_file = BufWriter::with_capacity(1 << 20, output_file);
    let mut writer = AsyncArrowWriter::try_new(output_file, merged_schema.clone(), None)?;

    let merge_start = Instant::now();
    stream_and_write_from_builder(builder1, &mut writer, &merged_schema, &mapping1).await?;
    stream_and_write_from_builder(builder2, &mut writer, &merged_schema, &mapping2).await?;
    writer.close().await?;

    let merge_duration = merge_start.elapsed();
    println!("\nParquet files merged with widening into async_merged_streaming_widened.parquet");
    print_parquet_contents(
        "async_merged_streaming_widened.parquet",
        "Contents of async_merged_streaming_widened.parquet",
    )
    .await?;

    println!("Merge operation took: {merge_duration:?}");
    Ok(())
}
