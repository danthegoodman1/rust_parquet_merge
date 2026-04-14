use std::fs::File as StdFile;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::{
    ArrayRef, Float64Array, Int32Array, Int64Array, ListArray, RecordBatch, StringArray,
    StructArray,
};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use parquet::arrow::async_writer::AsyncArrowWriter;
use parquet::file::properties::WriterProperties;
use rust_parquet_merge::{
    CompactionOptions, CompactionReport, PayloadMergeOptions, compact_ndjson_to_parquet,
    merge_payload_parquet_files,
};
use serde_json::{Map, Value, json};
use tokio::fs;

#[derive(Clone, Copy)]
enum SchemaFamily {
    Left,
    Right,
}

fn unique_benchmark_dir() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "rust_parquet_merge_payload_benchmark_{}_{}",
        std::process::id(),
        nonce
    ))
}

fn payload_batch(
    schema_family: SchemaFamily,
    rows: usize,
    start_event_id: i32,
) -> (SchemaRef, RecordBatch) {
    match schema_family {
        SchemaFamily::Left => {
            let profile_fields: Fields =
                vec![Arc::new(Field::new("name", DataType::Utf8, true))].into();
            let profile_array = Arc::new(StructArray::new(
                profile_fields.clone(),
                vec![Arc::new(StringArray::from(
                    (0..rows)
                        .map(|index| {
                            if index % 7 == 0 {
                                None
                            } else {
                                Some(format!("user_{start_event_id}_{index}"))
                            }
                        })
                        .collect::<Vec<_>>(),
                )) as ArrayRef],
                None,
            )) as ArrayRef;

            let scores =
                ListArray::from_iter_primitive::<Int32Type, _, _>((0..rows).map(|index| {
                    if index % 11 == 0 {
                        None
                    } else {
                        Some(vec![
                            Some(index as i32),
                            Some(index as i32 + 1),
                            Some(index as i32 + 2),
                        ])
                    }
                }));

            let payload_fields: Fields = vec![
                Arc::new(Field::new("score", DataType::Int32, true)),
                Arc::new(Field::new(
                    "profile",
                    DataType::Struct(profile_fields.clone()),
                    true,
                )),
                Arc::new(Field::new(
                    "scores",
                    DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                    true,
                )),
            ]
            .into();
            let payload_array = Arc::new(StructArray::new(
                payload_fields.clone(),
                vec![
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| Some((index % 100) as i32))
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    profile_array,
                    Arc::new(scores) as ArrayRef,
                ],
                None,
            )) as ArrayRef;

            let schema = Arc::new(Schema::new(vec![
                Field::new("event_id", DataType::Int32, false),
                Field::new("org_id", DataType::Int32, false),
                Field::new("payload", DataType::Struct(payload_fields), true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| start_event_id + index as i32)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| 1_000 + (index % 8) as i32)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    payload_array,
                ],
            )
            .unwrap();

            (schema, batch)
        }
        SchemaFamily::Right => {
            let profile_fields: Fields =
                vec![Arc::new(Field::new("tier", DataType::Utf8, true))].into();
            let profile_array = Arc::new(StructArray::new(
                profile_fields.clone(),
                vec![Arc::new(StringArray::from(
                    (0..rows)
                        .map(|index| match index % 3 {
                            0 => Some("gold".to_string()),
                            1 => Some("silver".to_string()),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                )) as ArrayRef],
                None,
            )) as ArrayRef;

            let scores =
                ListArray::from_iter_primitive::<Float64Type, _, _>((0..rows).map(|index| {
                    if index % 13 == 0 {
                        None
                    } else {
                        Some(vec![
                            Some(index as f64 * 1.25),
                            Some(index as f64 * 1.25 + 0.5),
                        ])
                    }
                }));

            let payload_fields: Fields = vec![
                Arc::new(Field::new("score", DataType::Float64, true)),
                Arc::new(Field::new(
                    "profile",
                    DataType::Struct(profile_fields.clone()),
                    true,
                )),
                Arc::new(Field::new(
                    "scores",
                    DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                    true,
                )),
                Arc::new(Field::new("amount", DataType::Int32, true)),
            ]
            .into();
            let payload_array = Arc::new(StructArray::new(
                payload_fields.clone(),
                vec![
                    Arc::new(Float64Array::from(
                        (0..rows)
                            .map(|index| Some(index as f64 * 0.75))
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    profile_array,
                    Arc::new(scores) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| Some(((index * 3) % 200) as i32))
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
                None,
            )) as ArrayRef;

            let schema = Arc::new(Schema::new(vec![
                Field::new("org_id", DataType::Int32, false),
                Field::new("event_id", DataType::Int32, false),
                Field::new("payload", DataType::Struct(payload_fields), true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| 2_000 + (index % 8) as i32)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| start_event_id + index as i32)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    payload_array,
                ],
            )
            .unwrap();

            (schema, batch)
        }
    }
}

fn payload_batch_with_int64_envelope(
    schema_family: SchemaFamily,
    rows: usize,
    start_event_id: i64,
) -> (SchemaRef, RecordBatch) {
    let (schema, batch) = payload_batch(schema_family, rows, start_event_id as i32);
    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut columns = Vec::with_capacity(schema.fields().len());

    for (field, column) in schema.fields().iter().zip(batch.columns()) {
        if field.name() == "event_id" || field.name() == "org_id" {
            let source = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("envelope columns are Int32");
            let converted = Arc::new(Int64Array::from(
                source
                    .iter()
                    .map(|value| value.map(i64::from))
                    .collect::<Vec<_>>(),
            )) as ArrayRef;
            fields.push(Field::new(
                field.name(),
                DataType::Int64,
                field.is_nullable(),
            ));
            columns.push(converted);
        } else {
            fields.push((**field).clone());
            columns.push(column.clone());
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    (schema, batch)
}

async fn write_parquet_file(
    path: &Path,
    schema: SchemaRef,
    batch: RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = tokio::fs::File::create(path).await?;
    let mut writer = AsyncArrowWriter::try_new(file, schema, Some(WriterProperties::new()))?;
    writer.write(&batch).await?;
    writer.close().await?;
    Ok(())
}

fn write_ndjson_file(
    path: &Path,
    rows: usize,
    start_event_id: i32,
    schema_family: SchemaFamily,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = StdFile::create(path)?;
    for index in 0..rows {
        let value = match schema_family {
            SchemaFamily::Left => json!({
                "event_id": start_event_id + index as i32,
                "org_id": 1_000 + (index % 8) as i32,
                "score": (index % 100) as i32,
                "profile": {
                    "name": format!("ndjson_user_{start_event_id}_{index}")
                },
                "scores": [index as i32, index as i32 + 1, index as i32 + 2]
            }),
            SchemaFamily::Right => json!({
                "event_id": start_event_id + index as i32,
                "org_id": 2_000 + (index % 8) as i32,
                "score": index as f64 * 0.75,
                "profile": {
                    "tier": if index % 2 == 0 { "gold" } else { "silver" }
                },
                "scores": [index as f64 * 1.25, index as f64 * 1.25 + 0.5],
                "amount": ((index * 3) % 200) as i32
            }),
        };
        writeln!(file, "{value}")?;
    }
    Ok(())
}

fn write_unique_shape_ndjson_file(
    path: &Path,
    rows: usize,
    start_event_id: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = StdFile::create(path)?;
    for index in 0..rows {
        let mut object = Map::new();
        object.insert("event_id".to_string(), json!(start_event_id + index as i32));
        object.insert("org_id".to_string(), json!(3_000 + (index % 8) as i32));
        object.insert(
            format!("metric_{}", start_event_id + index as i32),
            json!((index as i64) * 11),
        );
        object.insert(
            "profile".to_string(),
            json!({
                "name": format!("unique_user_{start_event_id}_{index}"),
                "bucket": format!("bucket_{}", index % 17),
            }),
        );
        writeln!(file, "{}", Value::Object(object))?;
    }
    Ok(())
}

fn print_report(label: &str, report: &CompactionReport) {
    let seconds = report.total_duration.as_secs_f64().max(f64::EPSILON);
    let rows_per_second = report.rows as f64 / seconds;
    let mb_per_second = report.input_bytes as f64 / 1_000_000.0 / seconds;
    let cache_lookups = report.planning_shape_cache_hits + report.planning_shape_cache_misses;
    let hit_rate = if cache_lookups == 0 {
        0.0
    } else {
        report.planning_shape_cache_hits as f64 / cache_lookups as f64
    };
    println!(
        "{label}: rows={}, input={:.2} MB, output={:.2} MB, total={:?}, planning={:?}, exec={:?}, sorting={:?}, rows/sec={:.0}, input MB/sec={:.2}, peak RSS={:.2} MB, planning_threads={}, unique_shapes={}, shape_cache_hits={}, shape_cache_misses={}, shape_cache_hit_rate={:.1}%",
        report.rows,
        report.input_bytes as f64 / 1_000_000.0,
        report.output_bytes as f64 / 1_000_000.0,
        report.total_duration,
        report.planning_duration,
        report.execution_duration,
        report.sorting_duration,
        rows_per_second,
        mb_per_second,
        report.peak_rss_bytes as f64 / 1_000_000.0,
        report.planning_threads_used,
        report.planning_unique_shapes,
        report.planning_shape_cache_hits,
        report.planning_shape_cache_misses,
        hit_rate * 100.0,
    );
}

fn ndjson_options(payload_merge_options: &PayloadMergeOptions) -> CompactionOptions {
    CompactionOptions {
        envelope_fields: vec!["event_id".to_string(), "org_id".to_string()],
        payload_column: "payload".to_string(),
        widening_options: payload_merge_options.widening_options,
        batch_rows: 4_096,
        ..CompactionOptions::default()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    const ROWS_PER_PARQUET_FILE: usize = 20_000;
    const PARQUET_FILE_COUNT: usize = 6;
    const NDJSON_ROWS_PER_FILE: usize = 25_000;
    const UNIQUE_SHAPE_ROWS_PER_FILE: usize = 2_000;

    let benchmark_dir = unique_benchmark_dir();
    fs::create_dir_all(&benchmark_dir).await?;

    let payload_merge_options = PayloadMergeOptions::default();
    let ndjson_options = ndjson_options(&payload_merge_options);

    let mut parquet_inputs = Vec::new();
    for file_index in 0..PARQUET_FILE_COUNT {
        let family = if file_index % 2 == 0 {
            SchemaFamily::Left
        } else {
            SchemaFamily::Right
        };
        let (schema, batch) = payload_batch(
            family,
            ROWS_PER_PARQUET_FILE,
            10_000 + (file_index * ROWS_PER_PARQUET_FILE) as i32,
        );
        let path = benchmark_dir.join(format!("payload_merge_input_{file_index}.parquet"));
        write_parquet_file(&path, schema, batch).await?;
        parquet_inputs.push(path);
    }

    let payload_merge_output = benchmark_dir.join("payload_merge_output.parquet");
    let payload_merge_report = merge_payload_parquet_files(
        &parquet_inputs,
        &payload_merge_output,
        &payload_merge_options,
    )
    .await?;
    print_report("Payload Parquet Merge", &payload_merge_report);

    let ndjson_left = benchmark_dir.join("payload_left.ndjson");
    let ndjson_right = benchmark_dir.join("payload_right.ndjson");
    write_ndjson_file(
        &ndjson_left,
        NDJSON_ROWS_PER_FILE,
        100_000,
        SchemaFamily::Left,
    )?;
    write_ndjson_file(
        &ndjson_right,
        NDJSON_ROWS_PER_FILE,
        200_000,
        SchemaFamily::Right,
    )?;

    let ndjson_output = benchmark_dir.join("payload_compacted.parquet");
    let ndjson_compaction_report = compact_ndjson_to_parquet(
        &[ndjson_left.clone(), ndjson_right.clone()],
        &ndjson_output,
        &ndjson_options,
    )
    .await?;
    print_report("NDJSON Compaction (Mixed Drift)", &ndjson_compaction_report);

    let sorted_ndjson_output = benchmark_dir.join("payload_compacted_sorted.parquet");
    let sorted_ndjson_report = compact_ndjson_to_parquet(
        &[ndjson_right.clone(), ndjson_left.clone()],
        &sorted_ndjson_output,
        &CompactionOptions {
            sort_field: Some("event_id".to_string()),
            ..ndjson_options.clone()
        },
    )
    .await?;
    print_report(
        "NDJSON Compaction (Mixed Drift, Sorted)",
        &sorted_ndjson_report,
    );

    let repeated_shape_left = benchmark_dir.join("payload_repeated_left.ndjson");
    let repeated_shape_right = benchmark_dir.join("payload_repeated_right.ndjson");
    write_ndjson_file(
        &repeated_shape_left,
        NDJSON_ROWS_PER_FILE,
        400_000,
        SchemaFamily::Left,
    )?;
    write_ndjson_file(
        &repeated_shape_right,
        NDJSON_ROWS_PER_FILE,
        500_000,
        SchemaFamily::Left,
    )?;
    let repeated_shape_output = benchmark_dir.join("payload_repeated_shape.parquet");
    let repeated_shape_report = compact_ndjson_to_parquet(
        &[repeated_shape_left, repeated_shape_right],
        &repeated_shape_output,
        &ndjson_options,
    )
    .await?;
    print_report("NDJSON Compaction (Repeated Shape)", &repeated_shape_report);

    let unique_shape_left = benchmark_dir.join("payload_unique_left.ndjson");
    let unique_shape_right = benchmark_dir.join("payload_unique_right.ndjson");
    write_unique_shape_ndjson_file(&unique_shape_left, UNIQUE_SHAPE_ROWS_PER_FILE, 600_000)?;
    write_unique_shape_ndjson_file(&unique_shape_right, UNIQUE_SHAPE_ROWS_PER_FILE, 700_000)?;
    let unique_shape_output = benchmark_dir.join("payload_unique_shape.parquet");
    let unique_shape_report = compact_ndjson_to_parquet(
        &[unique_shape_left, unique_shape_right],
        &unique_shape_output,
        &ndjson_options,
    )
    .await?;
    print_report(
        "NDJSON Compaction (Unique Shape Stress)",
        &unique_shape_report,
    );

    let (partner_schema, partner_batch) =
        payload_batch_with_int64_envelope(SchemaFamily::Right, NDJSON_ROWS_PER_FILE, 300_000);
    let partner_path = benchmark_dir.join("payload_merge_partner.parquet");
    write_parquet_file(&partner_path, partner_schema, partner_batch).await?;

    let ndjson_merge_output = benchmark_dir.join("payload_compaction_merged.parquet");
    let ndjson_merge_report = merge_payload_parquet_files(
        &[ndjson_output.clone(), partner_path],
        &ndjson_merge_output,
        &payload_merge_options,
    )
    .await?;
    print_report("Compacted NDJSON + Merge", &ndjson_merge_report);

    let end_to_end_seconds =
        ndjson_compaction_report.total_duration + ndjson_merge_report.total_duration;
    println!(
        "NDJSON end-to-end total: {:?} (compaction {:?} + merge {:?})",
        end_to_end_seconds,
        ndjson_compaction_report.total_duration,
        ndjson_merge_report.total_duration
    );
    println!("Benchmark artifacts: {}", benchmark_dir.display());

    Ok(())
}
