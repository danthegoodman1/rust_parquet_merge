use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::io::Read;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::{
    ArrayRef, Float64Array, Int32Array, Int64Array, ListArray, RecordBatch, StringArray,
    StructArray,
};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_writer::AsyncArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use rust_parquet_merge::{
    CompactionReport, NumericWideningMode, ParquetCompression, ParquetMergeExecutionOptions,
    PayloadMergeOptions, TopLevelMergeOptions, UnorderedMergeOrder,
    merge_payload_parquet_files_with_execution, merge_top_level_parquet_files_with_execution,
};
use serde::Serialize;
use tokio::fs::{self, File};
use tokio::process::Command;

const DEFAULT_FILE_COUNT: usize = 6;
const DEFAULT_ROWS_PER_FILE: usize = 20_000;
const DEFAULT_MEASURED_RUNS: usize = 5;
const DEFAULT_GENERATION_BATCH_ROWS: usize = 50_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Scenario {
    TopLevelPragmatic,
    NestedPayloadPragmatic,
    OrderedPayloadPragmatic,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::TopLevelPragmatic => "top_level_pragmatic",
            Self::NestedPayloadPragmatic => "nested_payload_pragmatic",
            Self::OrderedPayloadPragmatic => "ordered_payload_pragmatic",
        }
    }

    fn ordering_field(self) -> Option<&'static str> {
        match self {
            Self::OrderedPayloadPragmatic => Some("event_id"),
            _ => None,
        }
    }

    fn parse_many(value: &str) -> Result<Vec<Self>, String> {
        let normalized = value.trim();
        if normalized.is_empty() || normalized.eq_ignore_ascii_case("all") {
            return Ok(vec![
                Self::TopLevelPragmatic,
                Self::NestedPayloadPragmatic,
                Self::OrderedPayloadPragmatic,
            ]);
        }

        let mut scenarios = Vec::new();
        for part in normalized.split(',') {
            let scenario = match part.trim() {
                "top_level_pragmatic" => Self::TopLevelPragmatic,
                "nested_payload_pragmatic" => Self::NestedPayloadPragmatic,
                "ordered_payload_pragmatic" => Self::OrderedPayloadPragmatic,
                other => {
                    return Err(format!(
                        "unsupported RPM_BENCH_SCENARIO value `{other}`; expected `top_level_pragmatic`, `nested_payload_pragmatic`, `ordered_payload_pragmatic`, or `all`"
                    ));
                }
            };
            if !scenarios.contains(&scenario) {
                scenarios.push(scenario);
            }
        }

        if scenarios.is_empty() {
            Ok(vec![
                Self::TopLevelPragmatic,
                Self::NestedPayloadPragmatic,
                Self::OrderedPayloadPragmatic,
            ])
        } else {
            Ok(scenarios)
        }
    }
}

#[derive(Clone, Copy)]
enum SchemaFamily {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum BenchmarkUnorderedMergeOrder {
    Preserve,
    Interleaved,
}

impl BenchmarkUnorderedMergeOrder {
    fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            "preserve" => Ok(Self::Preserve),
            "interleaved" => Ok(Self::Interleaved),
            other => Err(format!(
                "unsupported RPM_BENCH_RUST_UNORDERED_ORDER value `{other}`; expected `preserve` or `interleaved`"
            )),
        }
    }

    fn execution_order(self) -> UnorderedMergeOrder {
        match self {
            Self::Preserve => UnorderedMergeOrder::PreserveInputOrder,
            Self::Interleaved => UnorderedMergeOrder::AllowInterleaved,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Preserve => "preserve",
            Self::Interleaved => "interleaved",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
enum BenchmarkCompression {
    Uncompressed,
    Snappy,
    Lz4Raw,
    Zstd { level: i32 },
}

impl BenchmarkCompression {
    fn parse(value: &str) -> Result<Self, String> {
        let value = value.trim().to_ascii_lowercase();
        match value.as_str() {
            "uncompressed" => Ok(Self::Uncompressed),
            "snappy" => Ok(Self::Snappy),
            "lz4_raw" => Ok(Self::Lz4Raw),
            value if value.starts_with("zstd:") => {
                let level = value["zstd:".len()..]
                    .parse::<i32>()
                    .map_err(|error| format!("failed parsing zstd level in `{value}`: {error}"))?;
                if !(1..=22).contains(&level) {
                    return Err(format!("zstd level must be between 1 and 22, got {level}"));
                }
                Ok(Self::Zstd { level })
            }
            other => Err(format!(
                "unsupported RPM_BENCH_RUST_COMPRESSION value `{other}`; expected `uncompressed`, `snappy`, `lz4_raw`, or `zstd:<level>`"
            )),
        }
    }

    fn execution_compression(self) -> ParquetCompression {
        match self {
            Self::Uncompressed => ParquetCompression::Uncompressed,
            Self::Snappy => ParquetCompression::Snappy,
            Self::Lz4Raw => ParquetCompression::Lz4Raw,
            Self::Zstd { level } => ParquetCompression::Zstd { level },
        }
    }

    fn label(self) -> String {
        match self {
            Self::Uncompressed => "uncompressed".to_string(),
            Self::Snappy => "snappy".to_string(),
            Self::Lz4Raw => "lz4_raw".to_string(),
            Self::Zstd { level } => format!("zstd:{level}"),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkConfig {
    file_count: usize,
    rows_per_file: usize,
    measured_runs: usize,
    generation_batch_rows: usize,
    target_input_bytes: Option<u64>,
    rust_parallelism: usize,
    rust_unordered_order: BenchmarkUnorderedMergeOrder,
    rust_compression: BenchmarkCompression,
    rust_dictionary_enabled: bool,
    rust_read_batch_size: Option<usize>,
    rust_output_batch_rows: Option<usize>,
    rust_output_row_group_rows: Option<usize>,
    rust_prefetch_batches_per_source: Option<usize>,
    exact_validation: bool,
    duckdb_threads: Option<usize>,
    duckdb_compression: Option<String>,
    scenarios: Vec<Scenario>,
}

#[derive(Debug)]
struct GeneratedScenarioData {
    input_paths: Vec<PathBuf>,
    rows_per_file: usize,
    input_bytes: u64,
}

#[derive(Clone, Copy)]
struct ScenarioBatchLayout {
    start_event_id: i64,
    event_step: i64,
    org_base: i64,
}

#[derive(Serialize)]
struct BenchmarkSummary {
    duckdb_version: String,
    benchmark_dir: String,
    config: BenchmarkConfig,
    scenarios: Vec<ScenarioSummary>,
}

#[derive(Serialize)]
struct ScenarioSummary {
    name: String,
    rows_per_file: usize,
    expected_rows: u64,
    input_bytes: u64,
    rust_resolved_parallelism: usize,
    rust_unordered_order: String,
    rust_compression: String,
    rust_dictionary_enabled: bool,
    duckdb_threads: Option<usize>,
    duckdb_compression: Option<String>,
    rust_output_bytes: u64,
    duckdb_output_bytes: u64,
    rust_parquet_metadata: Vec<ParquetMetadataSummary>,
    duckdb_parquet_metadata: Vec<ParquetMetadataSummary>,
    validation: ValidationSummary,
    rust: Option<EngineSummary>,
    duckdb: Option<EngineSummary>,
}

#[derive(Clone, Debug, Serialize)]
struct ParquetMetadataSummary {
    compression: String,
    encodings: String,
    chunks: u64,
    compressed_bytes: u64,
    uncompressed_bytes: u64,
}

#[derive(Serialize)]
struct ValidationSummary {
    status: String,
    rust_rows: u64,
    duckdb_rows: u64,
    rust_schema_shape: Vec<String>,
    duckdb_schema_shape: Vec<String>,
    rust_is_sorted: Option<bool>,
    duckdb_is_sorted: Option<bool>,
    exact_validation: bool,
    rust_minus_duckdb_rows: Option<u64>,
    duckdb_minus_rust_rows: Option<u64>,
    message: Option<String>,
}

#[derive(Serialize)]
struct OrderedMetricsSummary {
    ordered_merge_ms: u128,
    stats_fast_path_ms: u128,
    read_decode_ms: u128,
    source_prepare_ms: u128,
    ordered_output_assembly_ms: u128,
    ordered_output_selection_ms: u128,
    ordered_output_materialization_ms: u128,
    writer_write_ms: u128,
    writer_encode_work_ms: u128,
    writer_sink_ms: u128,
    writer_close_ms: u128,
    fast_path_batches: u64,
    fallback_batches: u64,
    direct_batch_writes: u64,
    accumulator_flushes: u64,
    accumulator_concat_flushes: u64,
    accumulator_interleave_flushes: u64,
    copy_candidate_row_groups: u64,
    copied_row_groups: u64,
    copied_rows: u64,
    copied_compressed_bytes: u64,
    row_group_copy_ms: u128,
}

#[derive(Serialize)]
struct EngineSummary {
    warmup_ms: u128,
    measured_ms: Vec<u128>,
    median_ms: u128,
    rows_per_sec: f64,
    input_mb_per_sec: f64,
    peak_rss_bytes: Option<u64>,
    user_cpu_ms: Option<u128>,
    system_cpu_ms: Option<u128>,
    total_cpu_ms: Option<u128>,
    cpu_percent: Option<f64>,
    ordered_metrics: Option<OrderedMetricsSummary>,
}

#[derive(Clone, Debug)]
struct RustRunResult {
    duration: Duration,
    report: CompactionReport,
    user_cpu_duration: Option<Duration>,
    system_cpu_duration: Option<Duration>,
}

#[derive(Clone, Debug)]
struct EngineRunResult {
    duration: Duration,
    peak_rss_bytes: Option<u64>,
    user_cpu_duration: Option<Duration>,
    system_cpu_duration: Option<Duration>,
}

#[derive(Clone, Copy, Debug)]
struct CpuUsageSnapshot {
    user: Duration,
    system: Duration,
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

fn parse_env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match env::var(name) {
        Ok(value) => Ok(value
            .parse::<usize>()
            .map_err(|error| format!("failed to parse {name}=`{value}` as usize: {error}"))?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("failed reading {name}: {error}").into()),
    }
}

fn parse_env_optional_usize(name: &str) -> Result<Option<usize>, Box<dyn Error>> {
    match env::var(name) {
        Ok(value) => Ok(Some(value.parse::<usize>().map_err(|error| {
            format!("failed to parse {name}=`{value}` as usize: {error}")
        })?)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("failed reading {name}: {error}").into()),
    }
}

fn parse_env_bool(name: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    match env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => Ok(true),
            "false" | "0" | "no" => Ok(false),
            _ => Err(format!("failed to parse {name}=`{value}` as bool").into()),
        },
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("failed reading {name}: {error}").into()),
    }
}

fn parse_duckdb_compression() -> Result<Option<String>, Box<dyn Error>> {
    match env::var("RPM_BENCH_DUCKDB_COMPRESSION") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            match normalized.as_str() {
                "snappy" | "uncompressed" | "zstd" | "lz4" => Ok(Some(normalized)),
                other => Err(format!(
                    "unsupported RPM_BENCH_DUCKDB_COMPRESSION value `{other}`; expected `snappy`, `uncompressed`, `zstd`, or `lz4`"
                )
                .into()),
            }
        }
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("failed reading RPM_BENCH_DUCKDB_COMPRESSION: {error}").into()),
    }
}

fn parse_env_gib(name: &str) -> Result<Option<u64>, Box<dyn Error>> {
    match env::var(name) {
        Ok(value) => {
            let gib = value
                .parse::<f64>()
                .map_err(|error| format!("failed to parse {name}=`{value}` as f64: {error}"))?;
            if gib <= 0.0 {
                return Err(format!("{name} must be > 0, got {gib}").into());
            }
            Ok(Some((gib * 1024.0 * 1024.0 * 1024.0) as u64))
        }
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("failed reading {name}: {error}").into()),
    }
}

fn load_config() -> Result<BenchmarkConfig, Box<dyn Error>> {
    let file_count = parse_env_usize("RPM_BENCH_FILE_COUNT", DEFAULT_FILE_COUNT)?;
    let rows_per_file = parse_env_usize("RPM_BENCH_ROWS_PER_FILE", DEFAULT_ROWS_PER_FILE)?;
    let measured_runs = parse_env_usize("RPM_BENCH_MEASURED_RUNS", DEFAULT_MEASURED_RUNS)?;
    let generation_batch_rows = parse_env_usize(
        "RPM_BENCH_GENERATION_BATCH_ROWS",
        DEFAULT_GENERATION_BATCH_ROWS,
    )?;
    let target_input_bytes = parse_env_gib("RPM_BENCH_TARGET_INPUT_GIB")?;
    let rust_parallelism = parse_env_usize("RPM_BENCH_RUST_PARALLELISM", 1)?;
    let rust_unordered_order = BenchmarkUnorderedMergeOrder::parse(
        &env::var("RPM_BENCH_RUST_UNORDERED_ORDER").unwrap_or_else(|_| "preserve".to_string()),
    )?;
    let rust_compression = BenchmarkCompression::parse(
        &env::var("RPM_BENCH_RUST_COMPRESSION").unwrap_or_else(|_| "uncompressed".to_string()),
    )?;
    let rust_dictionary_enabled = parse_env_bool("RPM_BENCH_RUST_DICTIONARY", true)?;
    let rust_read_batch_size = parse_env_optional_usize("RPM_BENCH_RUST_READ_BATCH_SIZE")?;
    let rust_output_batch_rows = parse_env_optional_usize("RPM_BENCH_RUST_OUTPUT_BATCH_ROWS")?;
    let rust_output_row_group_rows =
        parse_env_optional_usize("RPM_BENCH_RUST_OUTPUT_ROW_GROUP_ROWS")?;
    let rust_prefetch_batches_per_source =
        parse_env_optional_usize("RPM_BENCH_RUST_PREFETCH_BATCHES_PER_SOURCE")?;
    let exact_validation = parse_env_bool("RPM_BENCH_EXACT_VALIDATION", false)?;
    let duckdb_threads = parse_env_optional_usize("RPM_BENCH_DUCKDB_THREADS")?;
    let duckdb_compression = parse_duckdb_compression()?;
    let scenarios = Scenario::parse_many(
        &env::var("RPM_BENCH_SCENARIO").unwrap_or_else(|_| "all".to_string()),
    )?;

    if file_count == 0 {
        return Err("RPM_BENCH_FILE_COUNT must be > 0".into());
    }
    if rows_per_file == 0 {
        return Err("RPM_BENCH_ROWS_PER_FILE must be > 0".into());
    }
    if measured_runs == 0 {
        return Err("RPM_BENCH_MEASURED_RUNS must be > 0".into());
    }
    if generation_batch_rows == 0 {
        return Err("RPM_BENCH_GENERATION_BATCH_ROWS must be > 0".into());
    }
    if duckdb_threads == Some(0) {
        return Err("RPM_BENCH_DUCKDB_THREADS must be > 0 when set".into());
    }
    if rust_read_batch_size == Some(0) {
        return Err("RPM_BENCH_RUST_READ_BATCH_SIZE must be > 0 when set".into());
    }
    if rust_output_batch_rows == Some(0) {
        return Err("RPM_BENCH_RUST_OUTPUT_BATCH_ROWS must be > 0 when set".into());
    }
    if rust_output_row_group_rows == Some(0) {
        return Err("RPM_BENCH_RUST_OUTPUT_ROW_GROUP_ROWS must be > 0 when set".into());
    }
    if rust_prefetch_batches_per_source == Some(0) {
        return Err("RPM_BENCH_RUST_PREFETCH_BATCHES_PER_SOURCE must be > 0 when set".into());
    }

    Ok(BenchmarkConfig {
        file_count,
        rows_per_file,
        measured_runs,
        generation_batch_rows,
        target_input_bytes,
        rust_parallelism,
        rust_unordered_order,
        rust_compression,
        rust_dictionary_enabled,
        rust_read_batch_size,
        rust_output_batch_rows,
        rust_output_row_group_rows,
        rust_prefetch_batches_per_source,
        exact_validation,
        duckdb_threads,
        duckdb_compression,
        scenarios,
    })
}

fn top_level_batch(
    schema_family: SchemaFamily,
    rows: usize,
    start_event_id: i32,
    row_offset: usize,
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
                            .map(|index| start_event_id + (row_offset + index) as i32)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| {
                                let global_index = row_offset + index;
                                if global_index % 11 == 0 {
                                    None
                                } else {
                                    Some(((global_index * 3) % 100) as i32)
                                }
                            })
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(StringArray::from(
                        (0..rows)
                            .map(|index| {
                                let global_index = row_offset + index;
                                if global_index % 13 == 0 {
                                    None
                                } else {
                                    Some(format!("user_{}_{}", start_event_id, global_index))
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
                                let global_index = row_offset + index;
                                if global_index % 7 == 0 {
                                    None
                                } else {
                                    Some(global_index as f64 * 0.75)
                                }
                            })
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| start_event_id + (row_offset + index) as i32)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| {
                                let global_index = row_offset + index;
                                if global_index % 5 == 0 {
                                    None
                                } else {
                                    Some(20 + (global_index % 45) as i32)
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
    row_offset: usize,
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
                            let global_index = row_offset + index;
                            if global_index % 9 == 0 {
                                None
                            } else {
                                Some(format!("profile_{}_{}", start_event_id, global_index))
                            }
                        })
                        .collect::<Vec<_>>(),
                )) as ArrayRef],
                None,
            )) as ArrayRef;

            let scores =
                ListArray::from_iter_primitive::<Int32Type, _, _>((0..rows).map(|index| {
                    let global_index = row_offset + index;
                    if global_index % 17 == 0 {
                        None
                    } else {
                        Some(vec![
                            Some(global_index as i32),
                            Some(global_index as i32 + 1),
                            Some(global_index as i32 + 2),
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
                            .map(|index| Some(((row_offset + index) % 200) as i32))
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
                            .map(|index| start_event_id + (row_offset + index) as i32)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| 1_000 + ((row_offset + index) % 8) as i32)
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
                    let global_index = row_offset + index;
                    if global_index % 19 == 0 {
                        None
                    } else {
                        Some(vec![
                            Some(global_index as f64 * 1.25),
                            Some(global_index as f64 * 1.25 + 0.5),
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
                            .map(|index| Some((row_offset + index) as f64 * 0.5))
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    profile_array,
                    Arc::new(scores) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| Some((((row_offset + index) * 5) % 500) as i32))
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
                            .map(|index| 2_000 + ((row_offset + index) % 8) as i32)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| start_event_id + (row_offset + index) as i32)
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

fn ordered_payload_batch(
    schema_family: SchemaFamily,
    rows: usize,
    layout: ScenarioBatchLayout,
    row_offset: usize,
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
                            let global_index = row_offset + index;
                            if global_index % 9 == 0 {
                                None
                            } else {
                                Some(format!(
                                    "ordered_profile_{}_{}",
                                    layout.start_event_id, global_index
                                ))
                            }
                        })
                        .collect::<Vec<_>>(),
                )) as ArrayRef],
                None,
            )) as ArrayRef;

            let scores =
                ListArray::from_iter_primitive::<Int32Type, _, _>((0..rows).map(|index| {
                    let global_index = row_offset + index;
                    if global_index % 17 == 0 {
                        None
                    } else {
                        Some(vec![
                            Some(global_index as i32),
                            Some(global_index as i32 + 1),
                            Some(global_index as i32 + 2),
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
                            .map(|index| Some(((row_offset + index) % 200) as i32))
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
                            .map(|index| {
                                layout.start_event_id
                                    + ((row_offset + index) as i64 * layout.event_step)
                            })
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int64Array::from(
                        (0..rows)
                            .map(|index| layout.org_base + ((row_offset + index) % 8) as i64)
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
                        .map(|index| match (row_offset + index) % 3 {
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
                    let global_index = row_offset + index;
                    if global_index % 19 == 0 {
                        None
                    } else {
                        Some(vec![
                            Some(global_index as f64 * 1.25),
                            Some(global_index as f64 * 1.25 + 0.5),
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
                            .map(|index| Some((row_offset + index) as f64 * 0.75))
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    profile_array,
                    Arc::new(scores) as ArrayRef,
                    Arc::new(Int32Array::from(
                        (0..rows)
                            .map(|index| Some((((row_offset + index) * 5) % 500) as i32))
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
                            .map(|index| layout.org_base + ((row_offset + index) % 8) as i64)
                            .collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(Int64Array::from(
                        (0..rows)
                            .map(|index| {
                                layout.start_event_id
                                    + ((row_offset + index) as i64 * layout.event_step)
                            })
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

fn scenario_batch_layout(
    scenario: Scenario,
    file_index: usize,
    rows_per_file: usize,
    file_count: usize,
) -> ScenarioBatchLayout {
    match scenario {
        Scenario::OrderedPayloadPragmatic => ScenarioBatchLayout {
            start_event_id: file_index as i64,
            event_step: file_count as i64,
            org_base: 10_000 + ((file_index as i64) * 100),
        },
        _ => ScenarioBatchLayout {
            start_event_id: (file_index * rows_per_file) as i64,
            event_step: 1,
            org_base: 10_000 + ((file_index as i64) * 100),
        },
    }
}

fn scenario_batch(
    scenario: Scenario,
    schema_family: SchemaFamily,
    rows: usize,
    layout: ScenarioBatchLayout,
    row_offset: usize,
) -> (SchemaRef, RecordBatch) {
    match scenario {
        Scenario::TopLevelPragmatic => top_level_batch(
            schema_family,
            rows,
            layout.start_event_id as i32,
            row_offset,
        ),
        Scenario::NestedPayloadPragmatic => nested_payload_batch(
            schema_family,
            rows,
            layout.start_event_id as i32,
            row_offset,
        ),
        Scenario::OrderedPayloadPragmatic => {
            ordered_payload_batch(schema_family, rows, layout, row_offset)
        }
    }
}

async fn write_scenario_file(
    path: &Path,
    scenario: Scenario,
    schema_family: SchemaFamily,
    rows_per_file: usize,
    file_index: usize,
    file_count: usize,
    generation_batch_rows: usize,
) -> Result<(), Box<dyn Error>> {
    let layout = scenario_batch_layout(scenario, file_index, rows_per_file, file_count);
    let first_batch_rows = rows_per_file.min(generation_batch_rows);
    let (schema, first_batch) =
        scenario_batch(scenario, schema_family, first_batch_rows, layout, 0);
    let file = File::create(path).await?;
    let mut writer = AsyncArrowWriter::try_new(file, schema, Some(WriterProperties::new()))?;
    writer.write(&first_batch).await?;

    let mut row_offset = first_batch_rows;
    while row_offset < rows_per_file {
        let batch_rows = (rows_per_file - row_offset).min(generation_batch_rows);
        let (_, batch) = scenario_batch(scenario, schema_family, batch_rows, layout, row_offset);
        writer.write(&batch).await?;
        row_offset += batch_rows;
    }

    writer.close().await?;
    Ok(())
}

async fn estimate_rows_per_file(
    benchmark_dir: &Path,
    scenario: Scenario,
    config: &BenchmarkConfig,
) -> Result<usize, Box<dyn Error>> {
    let Some(target_input_bytes) = config.target_input_bytes else {
        return Ok(config.rows_per_file);
    };

    let calibration_dir = benchmark_dir
        .join(scenario.name())
        .join("calibration_inputs");
    fs::create_dir_all(&calibration_dir).await?;
    let calibration_rows = config.generation_batch_rows.max(10_000);
    let families = [SchemaFamily::Left, SchemaFamily::Right];
    let mut sample_bytes = 0_u64;

    for (index, family) in families.into_iter().enumerate() {
        let path = calibration_dir.join(format!("sample_{index}.parquet"));
        write_scenario_file(
            &path,
            scenario,
            family,
            calibration_rows,
            index,
            config.file_count,
            config.generation_batch_rows,
        )
        .await?;
        sample_bytes += std::fs::metadata(&path)?.len();
    }

    for entry in std::fs::read_dir(&calibration_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::remove_file(entry.path())?;
        }
    }

    let avg_bytes_per_row = sample_bytes as f64 / (calibration_rows * 2) as f64;
    if avg_bytes_per_row <= 0.0 {
        return Err("failed to estimate bytes per row for calibration".into());
    }

    let estimated_rows = ((target_input_bytes as f64)
        / (avg_bytes_per_row * config.file_count as f64))
        .ceil() as usize;

    Ok(estimated_rows.max(config.generation_batch_rows))
}

async fn generate_inputs(
    benchmark_dir: &Path,
    scenario: Scenario,
    config: &BenchmarkConfig,
) -> Result<GeneratedScenarioData, Box<dyn Error>> {
    let scenario_dir = benchmark_dir.join(scenario.name());
    fs::create_dir_all(&scenario_dir).await?;
    let rows_per_file = estimate_rows_per_file(benchmark_dir, scenario, config).await?;

    println!(
        "Generating {} inputs: files={}, rows/file={}, target_input_bytes={}",
        scenario.name(),
        config.file_count,
        rows_per_file,
        config
            .target_input_bytes
            .map(|bytes| bytes.to_string())
            .unwrap_or_else(|| "none".to_string())
    );

    let mut input_paths = Vec::with_capacity(config.file_count);
    for file_index in 0..config.file_count {
        let schema_family = if file_index % 2 == 0 {
            SchemaFamily::Left
        } else {
            SchemaFamily::Right
        };
        let path = scenario_dir.join(format!("input_{file_index}.parquet"));
        write_scenario_file(
            &path,
            scenario,
            schema_family,
            rows_per_file,
            file_index,
            config.file_count,
            config.generation_batch_rows,
        )
        .await?;
        input_paths.push(path);
    }

    let input_bytes = total_input_bytes(&input_paths)?;
    Ok(GeneratedScenarioData {
        input_paths,
        rows_per_file,
        input_bytes,
    })
}

fn total_input_bytes(input_paths: &[PathBuf]) -> Result<u64, Box<dyn Error>> {
    let mut total = 0_u64;
    for path in input_paths {
        total += std::fs::metadata(path)?.len();
    }
    Ok(total)
}

fn parquet_metadata_summary(path: &Path) -> Result<Vec<ParquetMetadataSummary>, Box<dyn Error>> {
    let file = std::fs::File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let mut groups = BTreeMap::<(String, String), ParquetMetadataSummary>::new();
    for row_group_index in 0..reader.num_row_groups() {
        let row_group = reader.metadata().row_group(row_group_index);
        for column in row_group.columns() {
            let compression = format!("{:?}", column.compression());
            let encodings = column
                .encodings()
                .iter()
                .map(|encoding| format!("{encoding:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            let entry = groups
                .entry((compression.clone(), encodings.clone()))
                .or_insert_with(|| ParquetMetadataSummary {
                    compression,
                    encodings,
                    chunks: 0,
                    compressed_bytes: 0,
                    uncompressed_bytes: 0,
                });
            entry.chunks += 1;
            entry.compressed_bytes += column.compressed_size() as u64;
            entry.uncompressed_bytes += column.uncompressed_size() as u64;
        }
    }
    Ok(groups.into_values().collect())
}

async fn parquet_row_count_and_schema(
    path: &Path,
    ordering_field: Option<&str>,
) -> Result<(u64, SchemaRef, Option<bool>), Box<dyn Error>> {
    let file = File::open(path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let schema = builder.schema().clone();
    let ordering_index = ordering_field
        .map(|field| {
            schema
                .index_of(field)
                .map_err(|error| format!("missing ordering field `{field}` in validation: {error}"))
        })
        .transpose()?;
    let mut stream = builder.build()?;
    let mut rows = 0_u64;
    let mut previous_event_id = None;
    let mut is_sorted = ordering_index.map(|_| true);
    while let Some(batch) = futures_util::StreamExt::next(&mut stream).await {
        let batch = batch?;
        rows += batch.num_rows() as u64;
        if let Some(index) = ordering_index {
            let event_ids = batch
                .column(index)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| "ordered validation expects Int64 ordering column".to_string())?;
            for value in event_ids.iter().flatten() {
                if let Some(previous) = previous_event_id {
                    if value < previous {
                        is_sorted = Some(false);
                    }
                }
                previous_event_id = Some(value);
            }
        }
    }
    Ok((rows, schema, is_sorted))
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

fn duckdb_merge_sql(
    scenario: Scenario,
    input_paths: &[PathBuf],
    output_path: &Path,
    duckdb_threads: Option<usize>,
    duckdb_compression: Option<&str>,
) -> String {
    let inputs = input_paths
        .iter()
        .map(|path| format!("'{}'", sql_quote(path)))
        .collect::<Vec<_>>()
        .join(", ");
    let select_sql = match scenario.ordering_field() {
        Some(field) => format!(
            "SELECT * FROM read_parquet([{inputs}], union_by_name = true) ORDER BY {field} NULLS LAST"
        ),
        None => format!("SELECT * FROM read_parquet([{inputs}], union_by_name = true)"),
    };
    let thread_pragma = duckdb_threads
        .map(|threads| format!("PRAGMA threads={threads}; "))
        .unwrap_or_default();
    let compression_clause = duckdb_compression
        .map(|compression| format!(", COMPRESSION '{compression}'"))
        .unwrap_or_default();
    format!(
        "{thread_pragma}COPY ({select_sql}) TO '{}' (FORMAT PARQUET{compression_clause});",
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

fn benchmark_execution_options(config: &BenchmarkConfig) -> ParquetMergeExecutionOptions {
    let mut options = ParquetMergeExecutionOptions {
        parallelism: config.rust_parallelism,
        unordered_merge_order: config.rust_unordered_order.execution_order(),
        writer_compression: config.rust_compression.execution_compression(),
        writer_dictionary_enabled: config.rust_dictionary_enabled,
        ..ParquetMergeExecutionOptions::default()
    };
    apply_benchmark_tuning(config, &mut options);
    options
}

fn apply_benchmark_tuning(config: &BenchmarkConfig, options: &mut ParquetMergeExecutionOptions) {
    if let Some(read_batch_size) = config.rust_read_batch_size {
        options.read_batch_size = read_batch_size;
    }
    if let Some(output_batch_rows) = config.rust_output_batch_rows {
        options.output_batch_rows = output_batch_rows;
    }
    if let Some(output_row_group_rows) = config.rust_output_row_group_rows {
        options.output_row_group_rows = output_row_group_rows;
    }
    if let Some(prefetch_batches_per_source) = config.rust_prefetch_batches_per_source {
        options.prefetch_batches_per_source = prefetch_batches_per_source;
    }
}

fn ordered_benchmark_execution_options(config: &BenchmarkConfig) -> ParquetMergeExecutionOptions {
    let mut options = ParquetMergeExecutionOptions {
        ordering_field: Some("event_id".to_string()),
        read_batch_size: 131_072,
        output_batch_rows: 131_072,
        prefetch_batches_per_source: 4,
        output_row_group_rows: 512_000,
        parallelism: config.rust_parallelism,
        unordered_merge_order: config.rust_unordered_order.execution_order(),
        writer_compression: config.rust_compression.execution_compression(),
        writer_dictionary_enabled: config.rust_dictionary_enabled,
        stats_fast_path: true,
    };
    apply_benchmark_tuning(config, &mut options);
    options
}

fn resolve_benchmark_parallelism(requested: usize, input_count: usize) -> usize {
    if input_count == 0 {
        return 0;
    }
    let target = if requested == 0 {
        std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
    } else {
        requested
    };
    target.max(1).min(input_count)
}

fn duration_from_timeval(value: libc::timeval) -> Duration {
    Duration::new(
        value.tv_sec.max(0) as u64,
        (value.tv_usec.max(0) as u32) * 1_000,
    )
}

fn peak_rss_bytes_from_rusage(usage: &libc::rusage) -> Option<u64> {
    if usage.ru_maxrss <= 0 {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        Some(usage.ru_maxrss as u64)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Some((usage.ru_maxrss as u64) * 1024)
    }
}

fn current_process_cpu_usage() -> Option<CpuUsageSnapshot> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    Some(CpuUsageSnapshot {
        user: duration_from_timeval(usage.ru_utime),
        system: duration_from_timeval(usage.ru_stime),
    })
}

fn duration_delta(after: Duration, before: Duration) -> Duration {
    after.checked_sub(before).unwrap_or_default()
}

fn cpu_usage_delta(
    before: Option<CpuUsageSnapshot>,
    after: Option<CpuUsageSnapshot>,
) -> (Option<Duration>, Option<Duration>) {
    match (before, after) {
        (Some(before), Some(after)) => (
            Some(duration_delta(after.user, before.user)),
            Some(duration_delta(after.system, before.system)),
        ),
        _ => (None, None),
    }
}

async fn run_rust_merge(
    scenario: Scenario,
    input_paths: &[PathBuf],
    output_path: &Path,
    config: &BenchmarkConfig,
) -> Result<RustRunResult, Box<dyn Error>> {
    let _ = fs::remove_file(output_path).await;
    let cpu_before = current_process_cpu_usage();
    let start = Instant::now();
    let report = match scenario {
        Scenario::TopLevelPragmatic => {
            merge_top_level_parquet_files_with_execution(
                input_paths,
                output_path,
                &TopLevelMergeOptions {
                    numeric_mode: NumericWideningMode::Float64Pragmatic,
                },
                &benchmark_execution_options(config),
            )
            .await?
        }
        Scenario::NestedPayloadPragmatic => {
            merge_payload_parquet_files_with_execution(
                input_paths,
                output_path,
                &PayloadMergeOptions::default(),
                &benchmark_execution_options(config),
            )
            .await?
        }
        Scenario::OrderedPayloadPragmatic => {
            merge_payload_parquet_files_with_execution(
                input_paths,
                output_path,
                &PayloadMergeOptions::default(),
                &ordered_benchmark_execution_options(config),
            )
            .await?
        }
    };
    let cpu_after = current_process_cpu_usage();
    let (user_cpu_duration, system_cpu_duration) = cpu_usage_delta(cpu_before, cpu_after);
    Ok(RustRunResult {
        duration: start.elapsed(),
        report,
        user_cpu_duration,
        system_cpu_duration,
    })
}

async fn run_duckdb_merge(
    scenario: Scenario,
    duckdb_bin: &str,
    input_paths: &[PathBuf],
    output_path: &Path,
    duckdb_threads: Option<usize>,
    duckdb_compression: Option<&str>,
) -> Result<EngineRunResult, Box<dyn Error>> {
    let _ = fs::remove_file(output_path).await;
    let sql = duckdb_merge_sql(
        scenario,
        input_paths,
        output_path,
        duckdb_threads,
        duckdb_compression,
    );
    let duckdb_bin = duckdb_bin.to_string();
    tokio::task::spawn_blocking(move || run_duckdb_merge_with_rusage(&duckdb_bin, &sql))
        .await
        .map_err(|error| format!("DuckDB merge task failed to join: {error}"))?
        .map_err(|error| error.to_string().into())
}

fn run_duckdb_merge_with_rusage(
    duckdb_bin: &str,
    sql: &str,
) -> Result<EngineRunResult, Box<dyn Error + Send + Sync>> {
    let mut child = std::process::Command::new(duckdb_bin)
        .arg("-c")
        .arg(sql)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    let pid = child.id() as libc::pid_t;
    let start = Instant::now();
    let mut status = 0_i32;
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let waited = unsafe { libc::wait4(pid, &mut status, 0, usage.as_mut_ptr()) };
    let duration = start.elapsed();
    if waited < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let usage = unsafe { usage.assume_init() };

    let mut stderr = String::new();
    if let Some(mut stderr_pipe) = child.stderr.take() {
        let _ = stderr_pipe.read_to_string(&mut stderr);
    }

    let exit_status = std::process::ExitStatus::from_raw(status);
    if !exit_status.success() {
        return Err(format!("DuckDB merge failed: {stderr}").into());
    }

    Ok(EngineRunResult {
        duration,
        peak_rss_bytes: peak_rss_bytes_from_rusage(&usage),
        user_cpu_duration: Some(duration_from_timeval(usage.ru_utime)),
        system_cpu_duration: Some(duration_from_timeval(usage.ru_stime)),
    })
}

fn duration_millis(duration: Duration) -> u128 {
    duration.as_millis()
}

fn median_duration(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn summarize_engine(
    warmup: &EngineRunResult,
    measured: &[EngineRunResult],
    rows: u64,
    input_bytes: u64,
    ordered_metrics: Option<OrderedMetricsSummary>,
) -> EngineSummary {
    let measured_durations = measured.iter().map(|run| run.duration).collect::<Vec<_>>();
    let median = median_duration(measured_durations.clone());
    let mut median_indices = (0..measured.len()).collect::<Vec<_>>();
    median_indices.sort_unstable_by_key(|index| measured[*index].duration);
    let median_run = &measured[median_indices[median_indices.len() / 2]];
    let seconds = median.as_secs_f64();
    let total_cpu_duration = median_run
        .user_cpu_duration
        .zip(median_run.system_cpu_duration)
        .map(|(user, system)| user + system);
    EngineSummary {
        warmup_ms: duration_millis(warmup.duration),
        measured_ms: measured_durations
            .iter()
            .map(|value| duration_millis(*value))
            .collect(),
        median_ms: duration_millis(median),
        rows_per_sec: rows as f64 / seconds,
        input_mb_per_sec: (input_bytes as f64 / (1024.0 * 1024.0)) / seconds,
        peak_rss_bytes: std::iter::once(warmup)
            .chain(measured.iter())
            .filter_map(|run| run.peak_rss_bytes)
            .max(),
        user_cpu_ms: median_run.user_cpu_duration.map(duration_millis),
        system_cpu_ms: median_run.system_cpu_duration.map(duration_millis),
        total_cpu_ms: total_cpu_duration.map(duration_millis),
        cpu_percent: total_cpu_duration.map(|cpu| cpu.as_secs_f64() / seconds * 100.0),
        ordered_metrics,
    }
}

fn ordered_metrics_from_report(report: &CompactionReport) -> Option<OrderedMetricsSummary> {
    if report.ordered_merge_duration == Duration::default()
        && report.stats_fast_path_duration == Duration::default()
        && report.ordered_output_assembly_duration == Duration::default()
        && report.ordered_output_selection_duration == Duration::default()
        && report.ordered_output_materialization_duration == Duration::default()
        && report.fast_path_batches == 0
        && report.fallback_batches == 0
        && report.direct_batch_writes == 0
        && report.accumulator_flushes == 0
        && report.copy_candidate_row_groups == 0
        && report.copied_row_groups == 0
    {
        return None;
    }

    Some(OrderedMetricsSummary {
        ordered_merge_ms: duration_millis(report.ordered_merge_duration),
        stats_fast_path_ms: duration_millis(report.stats_fast_path_duration),
        read_decode_ms: duration_millis(report.read_decode_duration),
        source_prepare_ms: duration_millis(report.source_prepare_duration),
        ordered_output_assembly_ms: duration_millis(report.ordered_output_assembly_duration),
        ordered_output_selection_ms: duration_millis(report.ordered_output_selection_duration),
        ordered_output_materialization_ms: duration_millis(
            report.ordered_output_materialization_duration,
        ),
        writer_write_ms: duration_millis(report.writer_write_duration),
        writer_encode_work_ms: duration_millis(report.writer_encode_duration),
        writer_sink_ms: duration_millis(report.writer_sink_duration),
        writer_close_ms: duration_millis(report.writer_close_duration),
        fast_path_batches: report.fast_path_batches,
        fallback_batches: report.fallback_batches,
        direct_batch_writes: report.direct_batch_writes,
        accumulator_flushes: report.accumulator_flushes,
        accumulator_concat_flushes: report.accumulator_concat_flushes,
        accumulator_interleave_flushes: report.accumulator_interleave_flushes,
        copy_candidate_row_groups: report.copy_candidate_row_groups,
        copied_row_groups: report.copied_row_groups,
        copied_rows: report.copied_rows,
        copied_compressed_bytes: report.copied_compressed_bytes,
        row_group_copy_ms: duration_millis(report.row_group_copy_duration),
    })
}

fn summarize_rust_runs(
    warmup: &RustRunResult,
    measured: &[RustRunResult],
    rows: u64,
    input_bytes: u64,
) -> EngineSummary {
    let mut median_indices = (0..measured.len()).collect::<Vec<_>>();
    median_indices.sort_unstable_by_key(|index| measured[*index].duration);
    let median_run = &measured[median_indices[median_indices.len() / 2]];
    let warmup_engine = EngineRunResult {
        duration: warmup.duration,
        peak_rss_bytes: Some(warmup.report.peak_rss_bytes),
        user_cpu_duration: warmup.user_cpu_duration,
        system_cpu_duration: warmup.system_cpu_duration,
    };
    let measured_engine = measured
        .iter()
        .map(|run| EngineRunResult {
            duration: run.duration,
            peak_rss_bytes: Some(run.report.peak_rss_bytes),
            user_cpu_duration: run.user_cpu_duration,
            system_cpu_duration: run.system_cpu_duration,
        })
        .collect::<Vec<_>>();
    summarize_engine(
        &warmup_engine,
        &measured_engine,
        rows,
        input_bytes,
        ordered_metrics_from_report(&median_run.report),
    )
}

async fn validate_outputs(
    scenario: Scenario,
    expected_rows: u64,
    rust_output: &Path,
    duckdb_output: &Path,
    duckdb_bin: &str,
    exact_validation: bool,
) -> Result<ValidationSummary, Box<dyn Error>> {
    let (rust_rows, rust_schema, rust_is_sorted) =
        parquet_row_count_and_schema(rust_output, scenario.ordering_field()).await?;
    let (duckdb_rows, duckdb_schema, duckdb_is_sorted) =
        parquet_row_count_and_schema(duckdb_output, scenario.ordering_field()).await?;

    let rust_shape = schema_shape(rust_schema.as_ref());
    let duckdb_shape = schema_shape(duckdb_schema.as_ref());

    let mut rust_minus_duckdb_rows = None;
    let mut duckdb_minus_rust_rows = None;
    if exact_validation {
        let (rust_minus, duckdb_minus) =
            duckdb_except_all_counts(duckdb_bin, rust_output, duckdb_output).await?;
        rust_minus_duckdb_rows = Some(rust_minus);
        duckdb_minus_rust_rows = Some(duckdb_minus);
    }

    let message = if rust_rows != expected_rows {
        Some(format!(
            "Rust output row count mismatch: expected {expected_rows}, got {rust_rows}"
        ))
    } else if duckdb_rows != expected_rows {
        Some(format!(
            "DuckDB output row count mismatch: expected {expected_rows}, got {duckdb_rows}"
        ))
    } else if rust_is_sorted == Some(false) {
        Some("Rust output is not sorted by event_id".to_string())
    } else if duckdb_is_sorted == Some(false) {
        Some("DuckDB output is not sorted by event_id".to_string())
    } else if rust_shape != duckdb_shape {
        Some("Rust and DuckDB output schemas differ".to_string())
    } else if rust_minus_duckdb_rows.unwrap_or(0) != 0 || duckdb_minus_rust_rows.unwrap_or(0) != 0 {
        Some(format!(
            "Rust and DuckDB exact output differs: rust_minus_duckdb={}, duckdb_minus_rust={}",
            rust_minus_duckdb_rows.unwrap_or(0),
            duckdb_minus_rust_rows.unwrap_or(0)
        ))
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
        rust_is_sorted,
        duckdb_is_sorted,
        exact_validation,
        rust_minus_duckdb_rows,
        duckdb_minus_rust_rows,
        message,
    })
}

fn duckdb_sql_string(value: &Path) -> String {
    let value = value.to_string_lossy();
    format!("'{}'", value.replace('\'', "''"))
}

async fn duckdb_except_all_counts(
    duckdb_bin: &str,
    rust_output: &Path,
    duckdb_output: &Path,
) -> Result<(u64, u64), Box<dyn Error>> {
    let rust = duckdb_sql_string(rust_output);
    let duckdb = duckdb_sql_string(duckdb_output);
    let sql = format!(
        "SELECT (SELECT count(*) FROM (SELECT * FROM read_parquet({rust}) EXCEPT ALL SELECT * FROM read_parquet({duckdb}))), (SELECT count(*) FROM (SELECT * FROM read_parquet({duckdb}) EXCEPT ALL SELECT * FROM read_parquet({rust})));"
    );
    let output = Command::new(duckdb_bin)
        .arg("-csv")
        .arg("-noheader")
        .arg("-c")
        .arg(sql)
        .output()
        .await?;
    if !output.status.success() {
        return Err(format!(
            "DuckDB exact validation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    let mut parts = stdout.trim().split(',');
    let rust_minus = parts
        .next()
        .ok_or("DuckDB exact validation returned no rust-minus count")?
        .parse::<u64>()?;
    let duckdb_minus = parts
        .next()
        .ok_or("DuckDB exact validation returned no duckdb-minus count")?
        .parse::<u64>()?;
    Ok((rust_minus, duckdb_minus))
}

async fn benchmark_scenario(
    benchmark_dir: &Path,
    duckdb_bin: &str,
    scenario: Scenario,
    config: &BenchmarkConfig,
) -> Result<ScenarioSummary, Box<dyn Error>> {
    let generated = generate_inputs(benchmark_dir, scenario, config).await?;
    let expected_rows = (config.file_count * generated.rows_per_file) as u64;
    let scenario_dir = benchmark_dir.join(scenario.name());

    let rust_warmup_output = scenario_dir.join("rust_warmup_output.parquet");
    let duckdb_warmup_output = scenario_dir.join("duckdb_warmup_output.parquet");
    let rust_warmup = run_rust_merge(
        scenario,
        &generated.input_paths,
        &rust_warmup_output,
        config,
    )
    .await?;
    let duckdb_warmup = run_duckdb_merge(
        scenario,
        duckdb_bin,
        &generated.input_paths,
        &duckdb_warmup_output,
        config.duckdb_threads,
        config.duckdb_compression.as_deref(),
    )
    .await?;

    let validation = validate_outputs(
        scenario,
        expected_rows,
        &rust_warmup_output,
        &duckdb_warmup_output,
        duckdb_bin,
        config.exact_validation,
    )
    .await?;
    let _ = fs::remove_file(&rust_warmup_output).await;
    let _ = fs::remove_file(&duckdb_warmup_output).await;
    if validation.status != "ok" {
        if let Some(message) = &validation.message {
            eprintln!("Validation failed for {}: {message}", scenario.name());
        }
        return Ok(ScenarioSummary {
            name: scenario.name().to_string(),
            rows_per_file: generated.rows_per_file,
            expected_rows,
            input_bytes: generated.input_bytes,
            rust_resolved_parallelism: resolve_benchmark_parallelism(
                config.rust_parallelism,
                generated.input_paths.len(),
            ),
            rust_unordered_order: config.rust_unordered_order.label().to_string(),
            rust_compression: config.rust_compression.label(),
            rust_dictionary_enabled: config.rust_dictionary_enabled,
            duckdb_threads: config.duckdb_threads,
            duckdb_compression: config.duckdb_compression.clone(),
            rust_output_bytes: 0,
            duckdb_output_bytes: 0,
            rust_parquet_metadata: Vec::new(),
            duckdb_parquet_metadata: Vec::new(),
            validation,
            rust: None,
            duckdb: None,
        });
    }

    let mut rust_runs = Vec::with_capacity(config.measured_runs);
    let mut duckdb_runs = Vec::with_capacity(config.measured_runs);
    let rust_output = scenario_dir.join("rust_measured_output.parquet");
    let duckdb_output = scenario_dir.join("duckdb_measured_output.parquet");

    for _ in 0..config.measured_runs {
        rust_runs
            .push(run_rust_merge(scenario, &generated.input_paths, &rust_output, config).await?);
    }
    for _ in 0..config.measured_runs {
        duckdb_runs.push(
            run_duckdb_merge(
                scenario,
                duckdb_bin,
                &generated.input_paths,
                &duckdb_output,
                config.duckdb_threads,
                config.duckdb_compression.as_deref(),
            )
            .await?,
        );
    }

    let rust_output_bytes = std::fs::metadata(&rust_output)?.len();
    let duckdb_output_bytes = std::fs::metadata(&duckdb_output)?.len();
    let rust_parquet_metadata = parquet_metadata_summary(&rust_output)?;
    let duckdb_parquet_metadata = parquet_metadata_summary(&duckdb_output)?;

    Ok(ScenarioSummary {
        name: scenario.name().to_string(),
        rows_per_file: generated.rows_per_file,
        expected_rows,
        input_bytes: generated.input_bytes,
        rust_resolved_parallelism: resolve_benchmark_parallelism(
            config.rust_parallelism,
            generated.input_paths.len(),
        ),
        rust_unordered_order: config.rust_unordered_order.label().to_string(),
        rust_compression: config.rust_compression.label(),
        rust_dictionary_enabled: config.rust_dictionary_enabled,
        duckdb_threads: config.duckdb_threads,
        duckdb_compression: config.duckdb_compression.clone(),
        rust_output_bytes,
        duckdb_output_bytes,
        rust_parquet_metadata,
        duckdb_parquet_metadata,
        validation,
        rust: Some(summarize_rust_runs(
            &rust_warmup,
            &rust_runs,
            expected_rows,
            generated.input_bytes,
        )),
        duckdb: Some(summarize_engine(
            &duckdb_warmup,
            &duckdb_runs,
            expected_rows,
            generated.input_bytes,
            None,
        )),
    })
}

fn format_metadata_summary(summary: &[ParquetMetadataSummary]) -> String {
    if summary.is_empty() {
        return "n/a".to_string();
    }
    summary
        .iter()
        .map(|entry| {
            format!(
                "{} [{}] chunks={} compressed={} uncompressed={}",
                entry.compression,
                entry.encodings,
                entry.chunks,
                entry.compressed_bytes,
                entry.uncompressed_bytes
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn format_optional_mib(bytes: Option<u64>) -> String {
    bytes
        .map(|bytes| format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0)))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_ms(value: Option<u128>) -> String {
    value
        .map(|value| format!("{value} ms"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_percent(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.0}%"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn print_summary(summary: &BenchmarkSummary) {
    println!("DuckDB CLI: {}", summary.duckdb_version);
    println!("Benchmark artifacts: {}", summary.benchmark_dir);
    println!(
        "Rust settings: parallelism={} unordered_order={} compression={} dictionary={} | DuckDB threads={} compression={}",
        summary.config.rust_parallelism,
        summary.config.rust_unordered_order.label(),
        summary.config.rust_compression.label(),
        summary.config.rust_dictionary_enabled,
        summary
            .config
            .duckdb_threads
            .map(|threads| threads.to_string())
            .unwrap_or_else(|| "default".to_string()),
        summary
            .config
            .duckdb_compression
            .as_deref()
            .unwrap_or("default")
    );
    for scenario in &summary.scenarios {
        println!();
        println!("Scenario: {}", scenario.name);
        println!(
            "  Inputs: rows/file={} total_input={:.2} MiB",
            scenario.rows_per_file,
            scenario.input_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  Rust execution: resolved_parallelism={} unordered_order={} compression={} dictionary={}",
            scenario.rust_resolved_parallelism,
            scenario.rust_unordered_order,
            scenario.rust_compression,
            scenario.rust_dictionary_enabled
        );
        println!(
            "  DuckDB execution: threads={} compression={}",
            scenario
                .duckdb_threads
                .map(|threads| threads.to_string())
                .unwrap_or_else(|| "default".to_string()),
            scenario.duckdb_compression.as_deref().unwrap_or("default")
        );
        println!(
            "  Validation: {} (rows rust={}, duckdb={})",
            scenario.validation.status,
            scenario.validation.rust_rows,
            scenario.validation.duckdb_rows
        );
        if let (Some(rust_is_sorted), Some(duckdb_is_sorted)) = (
            scenario.validation.rust_is_sorted,
            scenario.validation.duckdb_is_sorted,
        ) {
            println!(
                "  Order check: rust_sorted={} duckdb_sorted={}",
                rust_is_sorted, duckdb_is_sorted
            );
        }
        if scenario.validation.exact_validation {
            println!(
                "  Exact check: rust_minus_duckdb={} duckdb_minus_rust={}",
                scenario.validation.rust_minus_duckdb_rows.unwrap_or(0),
                scenario.validation.duckdb_minus_rust_rows.unwrap_or(0)
            );
        }
        if let Some(message) = &scenario.validation.message {
            println!("  Validation detail: {message}");
            continue;
        }
        println!(
            "  Output bytes: rust={} duckdb={}",
            scenario.rust_output_bytes, scenario.duckdb_output_bytes
        );
        println!(
            "  Rust parquet: {}",
            format_metadata_summary(&scenario.rust_parquet_metadata)
        );
        println!(
            "  DuckDB parquet: {}",
            format_metadata_summary(&scenario.duckdb_parquet_metadata)
        );

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
            "          peak_rss={} | cpu={} user={} sys={} ({})",
            format_optional_mib(rust.peak_rss_bytes),
            format_optional_ms(rust.total_cpu_ms),
            format_optional_ms(rust.user_cpu_ms),
            format_optional_ms(rust.system_cpu_ms),
            format_optional_percent(rust.cpu_percent),
        );
        if let Some(metrics) = &rust.ordered_metrics {
            println!(
                "  Rust ordered: merge={} ms | decode={} ms | prepare={} ms | assembly={} ms selection={} ms materialize={} ms | writer_elapsed={} ms encode_work={} ms sink={} ms close={} ms | fast_path={} ms",
                metrics.ordered_merge_ms,
                metrics.read_decode_ms,
                metrics.source_prepare_ms,
                metrics.ordered_output_assembly_ms,
                metrics.ordered_output_selection_ms,
                metrics.ordered_output_materialization_ms,
                metrics.writer_write_ms,
                metrics.writer_encode_work_ms,
                metrics.writer_sink_ms,
                metrics.writer_close_ms,
                metrics.stats_fast_path_ms,
            );
            println!(
                "                fallback_batches={} fast_path_batches={} direct_writes={} flushes={} concat_flushes={} interleave_flushes={}",
                metrics.fallback_batches,
                metrics.fast_path_batches,
                metrics.direct_batch_writes,
                metrics.accumulator_flushes,
                metrics.accumulator_concat_flushes,
                metrics.accumulator_interleave_flushes,
            );
            println!(
                "                copy_candidates={} copied_row_groups={} copied_rows={} copied_bytes={} copy_time={} ms",
                metrics.copy_candidate_row_groups,
                metrics.copied_row_groups,
                metrics.copied_rows,
                metrics.copied_compressed_bytes,
                metrics.row_group_copy_ms,
            );
        }
        println!(
            "  DuckDB median: {} ms | {:>10.0} rows/s | {:>8.2} MiB/s",
            duckdb.median_ms, duckdb.rows_per_sec, duckdb.input_mb_per_sec
        );
        println!(
            "          peak_rss={} | cpu={} user={} sys={} ({})",
            format_optional_mib(duckdb.peak_rss_bytes),
            format_optional_ms(duckdb.total_cpu_ms),
            format_optional_ms(duckdb.user_cpu_ms),
            format_optional_ms(duckdb.system_cpu_ms),
            format_optional_percent(duckdb.cpu_percent),
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config = load_config()?;
    let duckdb_bin = std::env::var("DUCKDB_BIN").unwrap_or_else(|_| "duckdb".to_string());
    let duckdb_version = ensure_duckdb_version(&duckdb_bin).await?;
    let benchmark_dir = unique_benchmark_dir();
    fs::create_dir_all(&benchmark_dir).await?;

    let mut scenarios = Vec::with_capacity(config.scenarios.len());
    for scenario in &config.scenarios {
        scenarios.push(benchmark_scenario(&benchmark_dir, &duckdb_bin, *scenario, &config).await?);
    }

    let summary = BenchmarkSummary {
        duckdb_version,
        benchmark_dir: benchmark_dir.display().to_string(),
        config,
        scenarios,
    };
    print_summary(&summary);

    let summary_path = benchmark_dir.join("rust_vs_duckdb_results.json");
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?).await?;
    println!("JSON summary: {}", summary_path.display());

    Ok(())
}
