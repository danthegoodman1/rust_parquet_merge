use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::{
    ArrayRef, Float64Array, Int32Array, ListArray, RecordBatch, StringArray, StructArray,
};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_writer::AsyncArrowWriter;
use parquet::file::properties::WriterProperties;
use rust_parquet_merge::{
    NumericWideningMode, PayloadMergeOptions, TopLevelMergeOptions, merge_payload_parquet_files,
    merge_top_level_parquet_files,
};
use serde::Serialize;
use tokio::fs::{self, File};
use tokio::process::Command;

const FILE_COUNT: usize = 6;
const ROWS_PER_FILE: usize = 20_000;
const MEASURED_RUNS: usize = 5;

#[derive(Clone, Copy, Debug)]
enum Scenario {
    TopLevelPragmatic,
    NestedPayloadPragmatic,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::TopLevelPragmatic => "top_level_pragmatic",
            Self::NestedPayloadPragmatic => "nested_payload_pragmatic",
        }
    }
}

#[derive(Clone, Copy)]
enum SchemaFamily {
    Left,
    Right,
}

#[derive(Clone, Copy)]
enum Engine {
    Rust,
    DuckDb,
}

#[derive(Serialize)]
struct BenchmarkSummary {
    duckdb_version: String,
    benchmark_dir: String,
    scenarios: Vec<ScenarioSummary>,
}

#[derive(Serialize)]
struct ScenarioSummary {
    name: String,
    expected_rows: u64,
    input_bytes: u64,
    validation: ValidationSummary,
    rust: Option<EngineSummary>,
    duckdb: Option<EngineSummary>,
}

#[derive(Serialize)]
struct ValidationSummary {
    status: String,
    rust_rows: u64,
    duckdb_rows: u64,
    rust_schema_shape: Vec<String>,
    duckdb_schema_shape: Vec<String>,
    message: Option<String>,
}

#[derive(Serialize)]
struct EngineSummary {
    warmup_ms: u128,
    measured_ms: Vec<u128>,
    median_ms: u128,
    rows_per_sec: f64,
    input_mb_per_sec: f64,
}

fn unique_benchmark_dir() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "rust_parquet_merge_duckdb_benchmark_{}_{}",
        std::process::id(),
        nonce
    ))
}

fn top_level_batch(
    schema_family: SchemaFamily,
    rows: usize,
    start_event_id: i32,
) -> (SchemaRef, RecordBatch) {
    match schema_family {
        SchemaFamily::Left => {
            let schema = Arc::new(Schema::new(vec![
                Field::new("event_id", DataType::Int32, false),
                Field::new("score", DataType::Int32, true),
                Field::new("name", DataType::Utf8, true),
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
                            .map(|index| {
                                if index % 11 == 0 {
                                    None
                                } else {
                                    Some(((index * 3) % 100) as i32)
                                }
                            })
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        (0..rows)
                            .map(|index| {
                                if index % 13 == 0 {
                                    None
                                } else {
                                    Some(format!("user_{start_event_id}_{index}"))
                                }
                            })
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
            )
            .unwrap();
            (schema, batch)
        }
        SchemaFamily::Right => {
            let schema = Arc::new(Schema::new(vec![
                Field::new("score", DataType::Float64, true),
                Field::new("event_id", DataType::Int32, false),
                Field::new("age", DataType::Int32, true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Float64Array::from(
                        (0..rows)
                            .map(|index| {
                                if index % 7 == 0 {
                                    None
                                } else {
                                    Some(index as f64 * 0.75)
                                }
                            })
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| start_event_id + index as i32)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| {
                                if index % 5 == 0 {
                                    None
                                } else {
                                    Some(20 + (index % 45) as i32)
                                }
                            })
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
            )
            .unwrap();
            (schema, batch)
        }
    }
}

fn nested_payload_batch(
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
                            if index % 9 == 0 {
                                None
                            } else {
                                Some(format!("profile_{start_event_id}_{index}"))
                            }
                        })
                        .collect::<Vec<_>>(),
                )) as ArrayRef],
                None,
            )) as ArrayRef;

            let scores =
                ListArray::from_iter_primitive::<Int32Type, _, _>((0..rows).map(|index| {
                    if index % 17 == 0 {
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
                            .map(|index| Some((index % 200) as i32))
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
                    if index % 19 == 0 {
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
                            .map(|index| Some(index as f64 * 0.5))
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    profile_array,
                    Arc::new(scores) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| Some(((index * 5) % 500) as i32))
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

async fn write_parquet_file(
    path: &Path,
    schema: SchemaRef,
    batch: RecordBatch,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path).await?;
    let mut writer = AsyncArrowWriter::try_new(file, schema, Some(WriterProperties::new()))?;
    writer.write(&batch).await?;
    writer.close().await?;
    Ok(())
}

async fn generate_inputs(
    benchmark_dir: &Path,
    scenario: Scenario,
) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let scenario_dir = benchmark_dir.join(scenario.name());
    fs::create_dir_all(&scenario_dir).await?;

    let mut input_paths = Vec::with_capacity(FILE_COUNT);
    for file_index in 0..FILE_COUNT {
        let schema_family = if file_index % 2 == 0 {
            SchemaFamily::Left
        } else {
            SchemaFamily::Right
        };
        let start_event_id = (file_index * ROWS_PER_FILE) as i32;
        let (schema, batch) = match scenario {
            Scenario::TopLevelPragmatic => {
                top_level_batch(schema_family, ROWS_PER_FILE, start_event_id)
            }
            Scenario::NestedPayloadPragmatic => {
                nested_payload_batch(schema_family, ROWS_PER_FILE, start_event_id)
            }
        };
        let path = scenario_dir.join(format!("input_{file_index}.parquet"));
        write_parquet_file(&path, schema, batch).await?;
        input_paths.push(path);
    }

    Ok(input_paths)
}

fn total_input_bytes(input_paths: &[PathBuf]) -> Result<u64, Box<dyn Error>> {
    let mut total = 0_u64;
    for path in input_paths {
        total += std::fs::metadata(path)?.len();
    }
    Ok(total)
}

async fn parquet_row_count_and_schema(path: &Path) -> Result<(u64, SchemaRef), Box<dyn Error>> {
    let file = File::open(path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let schema = builder.schema().clone();
    let mut stream = builder.build()?;
    let mut rows = 0_u64;
    while let Some(batch) = futures_util::StreamExt::next(&mut stream).await {
        rows += batch?.num_rows() as u64;
    }
    Ok((rows, schema))
}

fn data_type_shape(data_type: &DataType) -> String {
    match data_type {
        DataType::Struct(fields) => format!(
            "Struct({})",
            fields
                .iter()
                .map(|child| field_shape(child.as_ref()))
                .collect::<Vec<_>>()
                .join(",")
        ),
        DataType::List(child) => format!("List({})", data_type_shape(child.data_type())),
        DataType::LargeList(child) => format!("LargeList({})", data_type_shape(child.data_type())),
        other => format!("{other:?}"),
    }
}

fn field_shape(field: &Field) -> String {
    format!("{}:{}", field.name(), data_type_shape(field.data_type()))
}

fn schema_shape(schema: &Schema) -> Vec<String> {
    schema
        .fields()
        .iter()
        .map(|field| field_shape(field.as_ref()))
        .collect()
}

fn sql_quote(path: &Path) -> String {
    path.display().to_string().replace('\'', "''")
}

fn duckdb_merge_sql(input_paths: &[PathBuf], output_path: &Path) -> String {
    let inputs = input_paths
        .iter()
        .map(|path| format!("'{}'", sql_quote(path)))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "COPY (SELECT * FROM read_parquet([{inputs}], union_by_name = true)) TO '{}' (FORMAT PARQUET);",
        sql_quote(output_path)
    )
}

async fn ensure_duckdb_version(duckdb_bin: &str) -> Result<String, Box<dyn Error>> {
    let output = Command::new(duckdb_bin).arg("--version").output().await?;
    if !output.status.success() {
        return Err(format!(
            "failed to run `{duckdb_bin} --version`: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !version.contains("v1.4.1") {
        return Err(format!(
            "expected DuckDB CLI v1.4.1, got `{version}` from `{duckdb_bin} --version`"
        )
        .into());
    }

    Ok(version)
}

async fn run_rust_merge(
    scenario: Scenario,
    input_paths: &[PathBuf],
    output_path: &Path,
) -> Result<Duration, Box<dyn Error>> {
    let _ = fs::remove_file(output_path).await;
    let start = Instant::now();
    match scenario {
        Scenario::TopLevelPragmatic => {
            merge_top_level_parquet_files(
                input_paths,
                output_path,
                &TopLevelMergeOptions {
                    numeric_mode: NumericWideningMode::Float64Pragmatic,
                },
            )
            .await?;
        }
        Scenario::NestedPayloadPragmatic => {
            merge_payload_parquet_files(input_paths, output_path, &PayloadMergeOptions::default())
                .await?;
        }
    }
    Ok(start.elapsed())
}

async fn run_duckdb_merge(
    duckdb_bin: &str,
    input_paths: &[PathBuf],
    output_path: &Path,
) -> Result<Duration, Box<dyn Error>> {
    let _ = fs::remove_file(output_path).await;
    let sql = duckdb_merge_sql(input_paths, output_path);
    let start = Instant::now();
    let output = Command::new(duckdb_bin).arg("-c").arg(sql).output().await?;
    if !output.status.success() {
        return Err(format!(
            "DuckDB merge failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(start.elapsed())
}

fn duration_millis(duration: Duration) -> u128 {
    duration.as_millis()
}

fn median_duration(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn summarize_engine(
    warmup: Duration,
    measured: &[Duration],
    rows: u64,
    input_bytes: u64,
) -> EngineSummary {
    let median = median_duration(measured.to_vec());
    let seconds = median.as_secs_f64();
    EngineSummary {
        warmup_ms: duration_millis(warmup),
        measured_ms: measured
            .iter()
            .map(|value| duration_millis(*value))
            .collect(),
        median_ms: duration_millis(median),
        rows_per_sec: rows as f64 / seconds,
        input_mb_per_sec: (input_bytes as f64 / (1024.0 * 1024.0)) / seconds,
    }
}

async fn validate_outputs(
    expected_rows: u64,
    rust_output: &Path,
    duckdb_output: &Path,
) -> Result<ValidationSummary, Box<dyn Error>> {
    let (rust_rows, rust_schema) = parquet_row_count_and_schema(rust_output).await?;
    let (duckdb_rows, duckdb_schema) = parquet_row_count_and_schema(duckdb_output).await?;

    let rust_shape = schema_shape(rust_schema.as_ref());
    let duckdb_shape = schema_shape(duckdb_schema.as_ref());

    let message = if rust_rows != expected_rows {
        Some(format!(
            "Rust output row count mismatch: expected {expected_rows}, got {rust_rows}"
        ))
    } else if duckdb_rows != expected_rows {
        Some(format!(
            "DuckDB output row count mismatch: expected {expected_rows}, got {duckdb_rows}"
        ))
    } else if rust_shape != duckdb_shape {
        Some("Rust and DuckDB output schemas differ".to_string())
    } else {
        None
    };

    Ok(ValidationSummary {
        status: if message.is_some() {
            "failed".to_string()
        } else {
            "ok".to_string()
        },
        rust_rows,
        duckdb_rows,
        rust_schema_shape: rust_shape,
        duckdb_schema_shape: duckdb_shape,
        message,
    })
}

async fn benchmark_engine(
    engine: Engine,
    scenario: Scenario,
    duckdb_bin: &str,
    input_paths: &[PathBuf],
    output_path: &Path,
) -> Result<Duration, Box<dyn Error>> {
    match engine {
        Engine::Rust => run_rust_merge(scenario, input_paths, output_path).await,
        Engine::DuckDb => run_duckdb_merge(duckdb_bin, input_paths, output_path).await,
    }
}

async fn benchmark_scenario(
    benchmark_dir: &Path,
    duckdb_bin: &str,
    scenario: Scenario,
) -> Result<ScenarioSummary, Box<dyn Error>> {
    let input_paths = generate_inputs(benchmark_dir, scenario).await?;
    let input_bytes = total_input_bytes(&input_paths)?;
    let expected_rows = (FILE_COUNT * ROWS_PER_FILE) as u64;
    let scenario_dir = benchmark_dir.join(scenario.name());

    let rust_warmup_output = scenario_dir.join("rust_warmup_output.parquet");
    let duckdb_warmup_output = scenario_dir.join("duckdb_warmup_output.parquet");
    let rust_warmup = benchmark_engine(
        Engine::Rust,
        scenario,
        duckdb_bin,
        &input_paths,
        &rust_warmup_output,
    )
    .await?;
    let duckdb_warmup = benchmark_engine(
        Engine::DuckDb,
        scenario,
        duckdb_bin,
        &input_paths,
        &duckdb_warmup_output,
    )
    .await?;

    let validation =
        validate_outputs(expected_rows, &rust_warmup_output, &duckdb_warmup_output).await?;
    if validation.status != "ok" {
        if let Some(message) = &validation.message {
            eprintln!("Validation failed for {}: {message}", scenario.name());
        }
        return Ok(ScenarioSummary {
            name: scenario.name().to_string(),
            expected_rows,
            input_bytes,
            validation,
            rust: None,
            duckdb: None,
        });
    }

    let mut rust_runs = Vec::with_capacity(MEASURED_RUNS);
    let mut duckdb_runs = Vec::with_capacity(MEASURED_RUNS);
    let rust_output = scenario_dir.join("rust_measured_output.parquet");
    let duckdb_output = scenario_dir.join("duckdb_measured_output.parquet");

    for _ in 0..MEASURED_RUNS {
        rust_runs.push(
            benchmark_engine(
                Engine::Rust,
                scenario,
                duckdb_bin,
                &input_paths,
                &rust_output,
            )
            .await?,
        );
    }
    for _ in 0..MEASURED_RUNS {
        duckdb_runs.push(
            benchmark_engine(
                Engine::DuckDb,
                scenario,
                duckdb_bin,
                &input_paths,
                &duckdb_output,
            )
            .await?,
        );
    }

    Ok(ScenarioSummary {
        name: scenario.name().to_string(),
        expected_rows,
        input_bytes,
        validation,
        rust: Some(summarize_engine(
            rust_warmup,
            &rust_runs,
            expected_rows,
            input_bytes,
        )),
        duckdb: Some(summarize_engine(
            duckdb_warmup,
            &duckdb_runs,
            expected_rows,
            input_bytes,
        )),
    })
}

fn print_summary(summary: &BenchmarkSummary) {
    println!("DuckDB CLI: {}", summary.duckdb_version);
    println!("Benchmark artifacts: {}", summary.benchmark_dir);
    for scenario in &summary.scenarios {
        println!();
        println!("Scenario: {}", scenario.name);
        println!(
            "  Validation: {} (rows rust={}, duckdb={})",
            scenario.validation.status,
            scenario.validation.rust_rows,
            scenario.validation.duckdb_rows
        );
        if let Some(message) = &scenario.validation.message {
            println!("  Validation detail: {message}");
            continue;
        }

        let rust = scenario
            .rust
            .as_ref()
            .expect("validated scenarios have rust stats");
        let duckdb = scenario
            .duckdb
            .as_ref()
            .expect("validated scenarios have duckdb stats");
        println!(
            "  Rust   median: {} ms | {:>10.0} rows/s | {:>8.2} MiB/s",
            rust.median_ms, rust.rows_per_sec, rust.input_mb_per_sec
        );
        println!(
            "  DuckDB median: {} ms | {:>10.0} rows/s | {:>8.2} MiB/s",
            duckdb.median_ms, duckdb.rows_per_sec, duckdb.input_mb_per_sec
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let duckdb_bin = std::env::var("DUCKDB_BIN").unwrap_or_else(|_| "duckdb".to_string());
    let duckdb_version = ensure_duckdb_version(&duckdb_bin).await?;
    let benchmark_dir = unique_benchmark_dir();
    fs::create_dir_all(&benchmark_dir).await?;

    let scenarios = vec![
        benchmark_scenario(&benchmark_dir, &duckdb_bin, Scenario::TopLevelPragmatic).await?,
        benchmark_scenario(
            &benchmark_dir,
            &duckdb_bin,
            Scenario::NestedPayloadPragmatic,
        )
        .await?,
    ];

    let summary = BenchmarkSummary {
        duckdb_version,
        benchmark_dir: benchmark_dir.display().to_string(),
        scenarios,
    };
    print_summary(&summary);

    let summary_path = benchmark_dir.join("rust_vs_duckdb_results.json");
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?).await?;
    println!("JSON summary: {}", summary_path.display());

    Ok(())
}
