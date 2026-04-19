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
    CompactionReport, ParquetMergeExecutionOptions, PayloadMergeOptions,
    merge_payload_parquet_files, merge_payload_parquet_files_with_execution,
};
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
        "rust_parquet_merge_ordered_benchmark_{}_{}",
        std::process::id(),
        nonce
    ))
}

fn payload_batch(
    schema_family: SchemaFamily,
    rows: usize,
    start_event_id: i64,
    event_step: i64,
    org_base: i64,
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
                Field::new("event_id", DataType::Int64, false),
                Field::new("org_id", DataType::Int64, false),
                Field::new("payload", DataType::Struct(payload_fields), true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(
                        (0..rows)
                            .map(|index| start_event_id + (index as i64 * event_step))
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int64Array::from(
                        (0..rows)
                            .map(|index| org_base + (index % 8) as i64)
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
                Field::new("org_id", DataType::Int64, false),
                Field::new("event_id", DataType::Int64, false),
                Field::new("payload", DataType::Struct(payload_fields), true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(
                        (0..rows)
                            .map(|index| org_base + (index % 8) as i64)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int64Array::from(
                        (0..rows)
                            .map(|index| start_event_id + (index as i64 * event_step))
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

async fn write_parquet_file(
    path: &Path,
    schema: SchemaRef,
    batch: RecordBatch,
    input_row_group_rows: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = tokio::fs::File::create(path).await?;
    let writer_properties = WriterProperties::builder()
        .set_max_row_group_size(input_row_group_rows)
        .build();
    let mut writer = AsyncArrowWriter::try_new(file, schema, Some(writer_properties))?;
    writer.write(&batch).await?;
    writer.close().await?;
    Ok(())
}

fn print_report(label: &str, report: &CompactionReport) {
    let seconds = report.total_duration.as_secs_f64().max(f64::EPSILON);
    let rows_per_second = report.rows as f64 / seconds;
    let mb_per_second = report.input_bytes as f64 / 1_000_000.0 / seconds;
    let total_batch_segments = report.fast_path_batches + report.fallback_batches;
    let fast_path_hit_rate = if total_batch_segments == 0 {
        0.0
    } else {
        report.fast_path_batches as f64 / total_batch_segments as f64 * 100.0
    };
    println!(
        "{label}: rows={}, input={:.2} MB, output={:.2} MB, total={:?}, planning={:?}, exec={:?}, ordered_merge={:?}, stats_fast_path={:?}, rows/sec={:.0}, input MB/sec={:.2}, input_batches={}, output_batches={}, adapter_cache_hits={}, adapter_cache_misses={}, fast_path_row_groups={}, fast_path_batches={}, fallback_batches={}, fast_path_hit_rate={:.1}%, peak RSS={:.2} MB",
        report.rows,
        report.input_bytes as f64 / 1_000_000.0,
        report.output_bytes as f64 / 1_000_000.0,
        report.total_duration,
        report.planning_duration,
        report.execution_duration,
        report.ordered_merge_duration,
        report.stats_fast_path_duration,
        rows_per_second,
        mb_per_second,
        report.input_batches,
        report.output_batches,
        report.adapter_cache_hits,
        report.adapter_cache_misses,
        report.fast_path_row_groups,
        report.fast_path_batches,
        report.fallback_batches,
        fast_path_hit_rate,
        report.peak_rss_bytes as f64 / 1_000_000.0,
    );
}

async fn write_input_family(
    benchmark_dir: &Path,
    prefix: &str,
    file_count: usize,
    rows_per_file: usize,
    input_row_group_rows: usize,
    family_for_index: impl Fn(usize) -> SchemaFamily,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut paths = Vec::with_capacity(file_count);
    let event_step = file_count as i64;
    for file_index in 0..file_count {
        let family = family_for_index(file_index);
        let (schema, batch) = payload_batch(
            family,
            rows_per_file,
            file_index as i64,
            event_step,
            10_000 + ((file_index as i64) * 100),
        );
        let path = benchmark_dir.join(format!("{prefix}_{file_index}.parquet"));
        write_parquet_file(&path, schema, batch, input_row_group_rows).await?;
        paths.push(path);
    }
    Ok(paths)
}

async fn write_non_overlapping_input_family(
    benchmark_dir: &Path,
    prefix: &str,
    file_count: usize,
    rows_per_file: usize,
    input_row_group_rows: usize,
    family_for_index: impl Fn(usize) -> SchemaFamily,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut paths = Vec::with_capacity(file_count);
    for file_index in 0..file_count {
        let family = family_for_index(file_index);
        let (schema, batch) = payload_batch(
            family,
            rows_per_file,
            (file_index * rows_per_file) as i64,
            1,
            20_000 + ((file_index as i64) * 100),
        );
        let path = benchmark_dir.join(format!("{prefix}_{file_index}.parquet"));
        write_parquet_file(&path, schema, batch, input_row_group_rows).await?;
        paths.push(path);
    }
    Ok(paths)
}

async fn write_partially_overlapping_input_family(
    benchmark_dir: &Path,
    prefix: &str,
    file_count: usize,
    rows_per_file: usize,
    input_row_group_rows: usize,
    family_for_index: impl Fn(usize) -> SchemaFamily,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut paths = Vec::with_capacity(file_count);
    let overlap_stride = (rows_per_file / 2).max(1) as i64;
    for file_index in 0..file_count {
        let family = family_for_index(file_index);
        let (schema, batch) = payload_batch(
            family,
            rows_per_file,
            file_index as i64 * overlap_stride,
            1,
            30_000 + ((file_index as i64) * 100),
        );
        let path = benchmark_dir.join(format!("{prefix}_{file_index}.parquet"));
        write_parquet_file(&path, schema, batch, input_row_group_rows).await?;
        paths.push(path);
    }
    Ok(paths)
}

fn string_key_batch(rows: usize, start_hour: usize, org_base: i64) -> (SchemaRef, RecordBatch) {
    let payload_fields: Fields = vec![Arc::new(Field::new("score", DataType::Int32, true))].into();
    let payload = Arc::new(StructArray::new(
        payload_fields.clone(),
        vec![Arc::new(Int32Array::from(
            (0..rows)
                .map(|index| Some(index as i32))
                .collect::<Vec<_>>(),
        )) as ArrayRef],
        None,
    )) as ArrayRef;
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_time", DataType::Utf8, false),
        Field::new("org_id", DataType::Int64, false),
        Field::new("payload", DataType::Struct(payload_fields), true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(
                (0..rows)
                    .map(|index| Some(format!("ts-{:08}", start_hour + index)))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                (0..rows)
                    .map(|index| org_base + index as i64)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            payload,
        ],
    )
    .unwrap();
    (schema, batch)
}

async fn write_string_key_inputs(
    benchmark_dir: &Path,
    prefix: &str,
    file_count: usize,
    rows_per_file: usize,
    input_row_group_rows: usize,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut paths = Vec::with_capacity(file_count);
    for file_index in 0..file_count {
        let (schema, batch) = string_key_batch(
            rows_per_file,
            file_index * rows_per_file,
            40_000 + ((file_index as i64) * 100),
        );
        let path = benchmark_dir.join(format!("{prefix}_{file_index}.parquet"));
        write_parquet_file(&path, schema, batch, input_row_group_rows).await?;
        paths.push(path);
    }
    Ok(paths)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let benchmark_dir = unique_benchmark_dir();
    fs::create_dir_all(&benchmark_dir).await?;

    let payload_merge_options = PayloadMergeOptions::default();
    let ordered_execution = ParquetMergeExecutionOptions {
        ordering_field: Some("event_id".to_string()),
        ..ParquetMergeExecutionOptions::default()
    };

    let unordered_inputs = write_input_family(
        &benchmark_dir,
        "unordered_payload",
        6,
        20_000,
        20_000,
        |index| {
            if index % 2 == 0 {
                SchemaFamily::Left
            } else {
                SchemaFamily::Right
            }
        },
    )
    .await?;
    let unordered_output = benchmark_dir.join("unordered_payload_output.parquet");
    let unordered_report =
        merge_payload_parquet_files(&unordered_inputs, &unordered_output, &payload_merge_options)
            .await?;
    print_report("Unordered Payload Merge Baseline", &unordered_report);

    let ordered_identical_inputs = write_input_family(
        &benchmark_dir,
        "ordered_fully_overlapping",
        4,
        40_000,
        16_384,
        |_| SchemaFamily::Left,
    )
    .await?;
    let ordered_identical_output = benchmark_dir.join("ordered_fully_overlapping_output.parquet");
    let ordered_identical_report = merge_payload_parquet_files_with_execution(
        &ordered_identical_inputs,
        &ordered_identical_output,
        &payload_merge_options,
        &ordered_execution,
    )
    .await?;
    print_report(
        "Ordered Merge (Fully Overlapping Baseline)",
        &ordered_identical_report,
    );

    let ordered_alternating_inputs = write_input_family(
        &benchmark_dir,
        "ordered_alternating",
        6,
        30_000,
        16_384,
        |index| {
            if index % 2 == 0 {
                SchemaFamily::Left
            } else {
                SchemaFamily::Right
            }
        },
    )
    .await?;
    let ordered_alternating_output = benchmark_dir.join("ordered_alternating_output.parquet");
    let ordered_alternating_report = merge_payload_parquet_files_with_execution(
        &ordered_alternating_inputs,
        &ordered_alternating_output,
        &payload_merge_options,
        &ordered_execution,
    )
    .await?;
    print_report(
        "Ordered Merge (Alternating Drift)",
        &ordered_alternating_report,
    );

    let partial_overlap_inputs = write_partially_overlapping_input_family(
        &benchmark_dir,
        "ordered_partial_overlap",
        4,
        80_000,
        8_192,
        |index| {
            if index % 2 == 0 {
                SchemaFamily::Left
            } else {
                SchemaFamily::Right
            }
        },
    )
    .await?;
    let partial_overlap_output = benchmark_dir.join("ordered_partial_overlap_output.parquet");
    let partial_overlap_report = merge_payload_parquet_files_with_execution(
        &partial_overlap_inputs,
        &partial_overlap_output,
        &payload_merge_options,
        &ordered_execution,
    )
    .await?;
    print_report(
        "Ordered Merge (Partially Overlapping Row Groups)",
        &partial_overlap_report,
    );

    let non_overlapping_inputs = write_non_overlapping_input_family(
        &benchmark_dir,
        "ordered_non_overlapping",
        4,
        120_000,
        8_192,
        |index| {
            if index % 2 == 0 {
                SchemaFamily::Left
            } else {
                SchemaFamily::Right
            }
        },
    )
    .await?;
    let non_overlapping_output = benchmark_dir.join("ordered_non_overlapping_output.parquet");
    let non_overlapping_report = merge_payload_parquet_files_with_execution(
        &non_overlapping_inputs,
        &non_overlapping_output,
        &payload_merge_options,
        &ordered_execution,
    )
    .await?;
    print_report(
        "Ordered Merge (Non-Overlapping Huge Files)",
        &non_overlapping_report,
    );

    let string_key_inputs =
        write_string_key_inputs(&benchmark_dir, "ordered_string_keys", 4, 12_000, 4_096).await?;
    let string_key_output = benchmark_dir.join("ordered_string_keys_output.parquet");
    let string_key_report = merge_payload_parquet_files_with_execution(
        &string_key_inputs,
        &string_key_output,
        &payload_merge_options,
        &ParquetMergeExecutionOptions {
            ordering_field: Some("event_time".to_string()),
            ..ParquetMergeExecutionOptions::default()
        },
    )
    .await?;
    print_report("Ordered Merge (String Keys)", &string_key_report);

    let many_small_inputs = write_input_family(
        &benchmark_dir,
        "ordered_many_small",
        20,
        8_000,
        4_096,
        |_| SchemaFamily::Left,
    )
    .await?;
    let many_small_output = benchmark_dir.join("ordered_many_small_output.parquet");
    let many_small_report = merge_payload_parquet_files_with_execution(
        &many_small_inputs,
        &many_small_output,
        &payload_merge_options,
        &ordered_execution,
    )
    .await?;
    print_report("Ordered Merge (Many Smaller Files)", &many_small_report);

    println!("Benchmark artifacts: {}", benchmark_dir.display());
    Ok(())
}
