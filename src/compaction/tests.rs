use super::*;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arrow_array::{
    BooleanArray, Date32Array, Date64Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
    StringArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt64Array,
};
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde_json::{Value, json};

fn payload_schema_options() -> PayloadMergeOptions {
    PayloadMergeOptions::default()
}

fn compaction_options() -> CompactionOptions {
    CompactionOptions {
        envelope_fields: vec!["event_id".to_string(), "org_id".to_string()],
        payload_column: "payload".to_string(),
        widening_options: WideningOptions::default(),
        batch_rows: 2,
        scan_parallelism: Some(1),
        read_buffer_bytes: 1 << 20,
        sort_field: None,
        sort_max_rows_in_memory: 262_144,
    }
}

fn compaction_options_with_parallelism(parallelism: usize) -> CompactionOptions {
    let mut options = compaction_options();
    options.scan_parallelism = Some(parallelism);
    options
}

fn compaction_options_with_sort(
    sort_field: &str,
    sort_max_rows_in_memory: usize,
) -> CompactionOptions {
    let mut options = compaction_options();
    options.sort_field = Some(sort_field.to_string());
    options.sort_max_rows_in_memory = sort_max_rows_in_memory;
    options
}

fn ordered_execution_options(ordering_field: &str) -> ParquetMergeExecutionOptions {
    ParquetMergeExecutionOptions {
        ordering_field: Some(ordering_field.to_string()),
        read_batch_size: 2,
        output_batch_rows: 2,
        prefetch_batches_per_source: 1,
        output_row_group_rows: 2,
        parallelism: 1,
        unordered_merge_order: UnorderedMergeOrder::PreserveInputOrder,
        writer_compression: ParquetCompression::Uncompressed,
        writer_dictionary_enabled: true,
        stats_fast_path: true,
    }
}

fn unique_path(prefix: &str, extension: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "rust_parquet_merge_compaction_{prefix}_{}_{}.{}",
        std::process::id(),
        nonce,
        extension
    ))
}

fn sample_payload_inputs() -> (SchemaRef, RecordBatch, SchemaRef, RecordBatch) {
    let left_profile_fields: Fields =
        vec![Arc::new(Field::new("name", DataType::Utf8, true))].into();
    let left_profile = Arc::new(StructArray::new(
        left_profile_fields.clone(),
        vec![Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob")])) as ArrayRef],
        None,
    )) as ArrayRef;
    let left_scores = ListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(vec![
        Some(vec![Some(1), Some(2)]),
        None,
    ]);
    let left_payload_fields: Fields = vec![
        Arc::new(Field::new("score", DataType::Int32, true)),
        Arc::new(Field::new(
            "profile",
            DataType::Struct(left_profile_fields.clone()),
            true,
        )),
        Arc::new(Field::new(
            "scores",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )),
    ]
    .into();
    let left_payload = Arc::new(StructArray::new(
        left_payload_fields.clone(),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef,
            left_profile,
            Arc::new(left_scores) as ArrayRef,
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
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef,
            left_payload,
        ],
    )
    .unwrap();

    let right_profile_fields: Fields =
        vec![Arc::new(Field::new("tier", DataType::Utf8, true))].into();
    let right_profile = Arc::new(StructArray::new(
        right_profile_fields.clone(),
        vec![Arc::new(StringArray::from(vec![Some("gold"), None])) as ArrayRef],
        None,
    )) as ArrayRef;
    let right_scores =
        ListArray::from_iter_primitive::<arrow_array::types::Float64Type, _, _>(vec![
            Some(vec![Some(1.5), Some(2.5)]),
            Some(vec![Some(3.5)]),
        ]);
    let right_payload_fields: Fields = vec![
        Arc::new(Field::new("score", DataType::Float64, true)),
        Arc::new(Field::new(
            "profile",
            DataType::Struct(right_profile_fields.clone()),
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
    let right_payload = Arc::new(StructArray::new(
        right_payload_fields.clone(),
        vec![
            Arc::new(Float64Array::from(vec![Some(4.5), Some(5.5)])) as ArrayRef,
            right_profile,
            Arc::new(right_scores) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(7), Some(9)])) as ArrayRef,
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
            Arc::new(Int32Array::from(vec![3, 4])) as ArrayRef,
            right_payload,
        ],
    )
    .unwrap();

    (left_schema, left_batch, right_schema, right_batch)
}

fn promote_payload_envelope_to_int64(
    schema: SchemaRef,
    batch: RecordBatch,
    event_ids: &[i64],
    org_ids: &[i64],
) -> (SchemaRef, RecordBatch) {
    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut columns = Vec::with_capacity(schema.fields().len());

    for (field, column) in schema.fields().iter().zip(batch.columns()) {
        match field.name().as_str() {
            "event_id" => {
                fields.push(Field::new("event_id", DataType::Int64, field.is_nullable()));
                columns.push(Arc::new(Int64Array::from(event_ids.to_vec())) as ArrayRef);
            }
            "org_id" => {
                fields.push(Field::new("org_id", DataType::Int64, field.is_nullable()));
                columns.push(Arc::new(Int64Array::from(org_ids.to_vec())) as ArrayRef);
            }
            _ => {
                fields.push((**field).clone());
                columns.push(column.clone());
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    (schema, batch)
}

async fn write_parquet_with_properties(
    path: &Path,
    schema: SchemaRef,
    batch: RecordBatch,
    writer_properties: Option<WriterProperties>,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path).await?;
    let mut writer = AsyncArrowWriter::try_new(file, schema, writer_properties)?;
    writer.write(&batch).await?;
    writer.close().await?;
    Ok(())
}

async fn write_parquet(
    path: &Path,
    schema: SchemaRef,
    batch: RecordBatch,
) -> Result<(), Box<dyn Error>> {
    write_parquet_with_properties(path, schema, batch, None).await
}

async fn read_parquet_batches(path: &Path) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
    let file = File::open(path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let mut stream = builder.build()?;
    let mut batches = Vec::new();
    while let Some(batch_result) = stream.next().await {
        batches.push(batch_result?);
    }
    Ok(batches)
}

fn parquet_compressions(path: &Path) -> Result<Vec<Compression>, Box<dyn Error>> {
    let file = StdFile::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let mut compressions = Vec::new();
    for row_group_index in 0..reader.num_row_groups() {
        let row_group = reader.metadata().row_group(row_group_index);
        for column in row_group.columns() {
            compressions.push(column.compression());
        }
    }
    Ok(compressions)
}

fn parquet_row_group_count(path: &Path) -> Result<usize, Box<dyn Error>> {
    let file = StdFile::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    Ok(reader.num_row_groups())
}

fn write_ndjson(path: &Path, values: &[Value]) -> Result<(), Box<dyn Error>> {
    let mut file = StdFile::create(path)?;
    for value in values {
        writeln!(file, "{value}")?;
    }
    Ok(())
}

#[test]
fn identical_schema_sources_get_fast_path_adapters() {
    let (left_schema, _, _, _) = sample_payload_inputs();
    let mut plan = CompiledPayloadPlan::new(left_schema.clone());
    let adapter = plan
        .source_adapter_for_schema(left_schema.as_ref())
        .unwrap();

    assert!(adapter.identical_schema);
    assert_eq!(plan.source_adapter_cache.len(), 1);
}

#[test]
fn source_adapter_cache_reuses_compiled_schema_plans() {
    let (left_schema, left_batch, right_schema, _) = sample_payload_inputs();
    let mut plan = build_compiled_payload_plan(
        [left_schema.as_ref(), right_schema.as_ref()],
        &payload_schema_options(),
    )
    .unwrap();

    let _ = plan.adapt_batch(&left_batch).unwrap();
    assert_eq!(plan.source_adapter_cache.len(), 1);
    let _ = plan.adapt_batch(&left_batch).unwrap();
    assert_eq!(plan.source_adapter_cache.len(), 1);
}

#[test]
fn nested_struct_and_list_adapters_precompute_child_indices() {
    let (left_schema, _, right_schema, _) = sample_payload_inputs();
    let mut plan = build_compiled_payload_plan(
        [left_schema.as_ref(), right_schema.as_ref()],
        &payload_schema_options(),
    )
    .unwrap();
    let adapter = plan
        .source_adapter_for_schema(left_schema.as_ref())
        .unwrap();

    let payload_op = adapter
        .operations
        .iter()
        .find(|operation| operation.target_field.name() == "payload")
        .expect("payload column is compiled");
    let CompiledTypeAdapter::StructAdapter(payload_adapter) = &payload_op.adapter else {
        panic!("payload operation must be a struct adapter");
    };

    assert_eq!(payload_adapter.children[0].source_index, Some(0));
    assert_eq!(payload_adapter.children[1].source_index, Some(1));
    assert_eq!(payload_adapter.children[2].source_index, Some(2));
    assert_eq!(payload_adapter.children[3].source_index, None);

    let CompiledTypeAdapter::StructAdapter(profile_adapter) = &payload_adapter.children[1].adapter
    else {
        panic!("payload.profile should compile to a nested struct adapter");
    };
    assert_eq!(profile_adapter.children[0].source_index, Some(0));
    assert_eq!(profile_adapter.children[1].source_index, None);

    let CompiledTypeAdapter::ListAdapter(scores_adapter) = &payload_adapter.children[2].adapter
    else {
        panic!("payload.scores should compile to a list adapter");
    };
    assert_eq!(scores_adapter.source_kind, ListKind::List);
}

#[test]
fn compiled_adapter_rejects_primitive_vs_struct_collisions() {
    let error = compile_type_adapter(
        &DataType::Int32,
        &DataType::Struct(vec![Arc::new(Field::new("x", DataType::Utf8, true))].into()),
    )
    .unwrap_err();

    assert!(error.contains("unsupported coercion"));
    assert!(error.contains("Int32"));
    assert!(error.contains("Struct"));
}

#[test]
fn discover_ndjson_schema_extracts_envelope_and_unions_payload() -> Result<(), Box<dyn Error>> {
    let left_path = unique_path("discover_left", "ndjson");
    let right_path = unique_path("discover_right", "ndjson");
    write_ndjson(
        &left_path,
        &[json!({
            "event_id": 1,
            "org_id": 10,
            "score": 1,
            "profile": { "name": "Alice" },
            "scores": [1, 2]
        })],
    )?;
    write_ndjson(
        &right_path,
        &[json!({
            "event_id": 2,
            "org_id": 20,
            "score": 4.5,
            "profile": { "tier": "gold" },
            "scores": [1.5],
            "amount": 9
        })],
    )?;

    let schema = discover_ndjson_schema_from_paths(
        &[left_path.clone(), right_path.clone()],
        &compaction_options(),
    )?;

    assert_eq!(schema.field(0).name(), "event_id");
    assert_eq!(schema.field(1).name(), "org_id");
    assert_eq!(schema.field(2).name(), "payload");

    let payload = schema.field_with_name("payload")?;
    let DataType::Struct(payload_fields) = payload.data_type() else {
        panic!("payload should be a struct");
    };
    let score_field = payload_fields
        .iter()
        .find(|field| field.name() == "score")
        .expect("payload.score exists");
    assert_eq!(score_field.data_type(), &DataType::Float64);
    let profile_field = payload_fields
        .iter()
        .find(|field| field.name() == "profile")
        .expect("payload.profile exists");
    let scores_field = payload_fields
        .iter()
        .find(|field| field.name() == "scores")
        .expect("payload.scores exists");
    let amount_field = payload_fields
        .iter()
        .find(|field| field.name() == "amount")
        .expect("payload.amount exists");
    assert_eq!(scores_field.name(), "scores");
    assert_eq!(amount_field.name(), "amount");

    let DataType::Struct(profile_fields) = profile_field.data_type() else {
        panic!("payload.profile should be a struct");
    };
    assert_eq!(profile_fields[0].name(), "name");
    assert_eq!(profile_fields[1].name(), "tier");

    std::fs::remove_file(left_path)?;
    std::fs::remove_file(right_path)?;
    Ok(())
}

#[test]
fn discover_ndjson_schema_rejects_structural_conflicts() -> Result<(), Box<dyn Error>> {
    let left_path = unique_path("discover_conflict_left", "ndjson");
    let right_path = unique_path("discover_conflict_right", "ndjson");
    write_ndjson(
        &left_path,
        &[json!({
            "event_id": 1,
            "org_id": 10,
            "profile": 1
        })],
    )?;
    write_ndjson(
        &right_path,
        &[json!({
            "event_id": 2,
            "org_id": 20,
            "profile": { "name": "Alice" }
        })],
    )?;

    let error = discover_ndjson_schema_from_paths(
        &[left_path.clone(), right_path.clone()],
        &compaction_options(),
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("payload.profile"));
    assert!(error.contains("Int64"));
    assert!(error.contains("Struct"));

    std::fs::remove_file(left_path)?;
    std::fs::remove_file(right_path)?;
    Ok(())
}

#[test]
fn discover_ndjson_shape_cache_matches_uncached_schema() -> Result<(), Box<dyn Error>> {
    let left_path = unique_path("discover_cache_left", "ndjson");
    let right_path = unique_path("discover_cache_right", "ndjson");
    write_ndjson(
        &left_path,
        &[
            json!({
                "event_id": 1,
                "org_id": 10,
                "score": 1,
                "profile": { "name": "Alice" },
                "scores": [1, 2]
            }),
            json!({
                "event_id": 2,
                "org_id": 10,
                "score": 2,
                "profile": { "name": "Bob" },
                "scores": [3, 4]
            }),
        ],
    )?;
    write_ndjson(
        &right_path,
        &[
            json!({
                "event_id": 3,
                "org_id": 20,
                "score": 4.5,
                "profile": { "tier": "gold" },
                "scores": [1.5],
                "amount": 9
            }),
            json!({
                "event_id": 4,
                "org_id": 20,
                "score": 5.5,
                "profile": { "tier": "silver" },
                "scores": [2.5],
                "amount": 11
            }),
        ],
    )?;

    let cached = discover_ndjson_schema_details_with_settings(
        &[left_path.clone(), right_path.clone()],
        &compaction_options(),
        true,
    )?;
    let uncached = discover_ndjson_schema_details_with_settings(
        &[left_path.clone(), right_path.clone()],
        &compaction_options(),
        false,
    )?;

    assert_eq!(cached.schema, uncached.schema);
    assert!(cached.stats.shape_cache_hits > 0);
    assert_eq!(uncached.stats.shape_cache_hits, 0);
    assert!(cached.stats.shape_cache_misses > 0);

    std::fs::remove_file(left_path)?;
    std::fs::remove_file(right_path)?;
    Ok(())
}

#[test]
fn discover_ndjson_parallel_reduction_is_deterministic() -> Result<(), Box<dyn Error>> {
    let left_path = unique_path("discover_parallel_left", "ndjson");
    let right_path = unique_path("discover_parallel_right", "ndjson");
    write_ndjson(
        &left_path,
        &[
            json!({
                "event_id": 1,
                "org_id": 10,
                "score": 1,
                "profile": { "name": "Alice" },
                "scores": [1, 2]
            }),
            json!({
                "event_id": 2,
                "org_id": 10,
                "score": 2,
                "profile": { "name": "Bob" },
                "scores": [3, 4],
                "tags": ["a", "b"]
            }),
        ],
    )?;
    write_ndjson(
        &right_path,
        &[
            json!({
                "event_id": 3,
                "org_id": 20,
                "score": 4.5,
                "profile": { "tier": "gold" },
                "scores": [1.5],
                "amount": 9
            }),
            json!({
                "event_id": 4,
                "org_id": 20,
                "score": 5.5,
                "profile": { "tier": "silver" },
                "scores": [2.5],
                "amount": 11,
                "tags": ["c"]
            }),
        ],
    )?;

    let single = discover_ndjson_schema_from_paths(
        &[left_path.clone(), right_path.clone()],
        &compaction_options_with_parallelism(1),
    )?;
    let parallel = discover_ndjson_schema_from_paths(
        &[left_path.clone(), right_path.clone()],
        &compaction_options_with_parallelism(2),
    )?;
    let reversed = discover_ndjson_schema_from_paths(
        &[right_path.clone(), left_path.clone()],
        &compaction_options_with_parallelism(2),
    )?;

    assert_eq!(single, parallel);
    assert_eq!(single, reversed);

    std::fs::remove_file(left_path)?;
    std::fs::remove_file(right_path)?;
    Ok(())
}

#[test]
fn compiled_ndjson_plan_null_fills_missing_payload_fields() -> Result<(), Box<dyn Error>> {
    let payload_fields: Fields = vec![
        Arc::new(Field::new("amount", DataType::Int64, true)),
        Arc::new(Field::new(
            "profile",
            DataType::Struct(
                vec![
                    Arc::new(Field::new("name", DataType::Utf8, true)),
                    Arc::new(Field::new("tier", DataType::Utf8, true)),
                ]
                .into(),
            ),
            true,
        )),
        Arc::new(Field::new("score", DataType::Float64, true)),
        Arc::new(Field::new(
            "scores",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        )),
    ]
    .into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int64, true),
        Field::new("org_id", DataType::Int64, true),
        Field::new("payload", DataType::Struct(payload_fields.clone()), true),
    ]));
    let plan = CompiledNdjsonPlan::compile(schema.as_ref(), &compaction_options())?;
    let mut batch_builder = NdjsonBatchBuilder::new(schema.clone(), plan, 8)?;

    let mut first_line =
        br#"{"event_id":1,"org_id":10,"score":1,"profile":{"name":"Alice"},"scores":[1,2]}"#
            .to_vec();
    let mut first_buffers = Buffers::default();
    let first_value =
        to_borrowed_value_with_buffers(first_line.as_mut_slice(), &mut first_buffers)?;
    batch_builder.append_record(&first_value)?;

    let mut second_line =
        br#"{"event_id":2,"org_id":20,"score":4.5,"amount":9,"profile":{"tier":"gold"}}"#.to_vec();
    let mut second_buffers = Buffers::default();
    let second_value =
        to_borrowed_value_with_buffers(second_line.as_mut_slice(), &mut second_buffers)?;
    batch_builder.append_record(&second_value)?;

    let batch = batch_builder.finish_batch();
    let event_ids = batch
        .column(batch.schema().index_of("event_id")?)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("event_id is Int64");
    assert_eq!(event_ids.value(0), 1);
    assert_eq!(event_ids.value(1), 2);

    let payload = batch
        .column(batch.schema().index_of("payload")?)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("payload is Struct");
    let score_array = payload
        .column_by_name("score")
        .expect("score exists")
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("score is Float64");
    assert_eq!(score_array.value(0), 1.0);
    assert_eq!(score_array.value(1), 4.5);

    let amount_array = payload
        .column_by_name("amount")
        .expect("amount exists")
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("amount is Int64");
    assert!(amount_array.is_null(0));
    assert_eq!(amount_array.value(1), 9);

    let profile = payload
        .column_by_name("profile")
        .expect("profile exists")
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("profile is Struct");
    let name_array = profile
        .column_by_name("name")
        .expect("name exists")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("name is Utf8");
    let tier_array = profile
        .column_by_name("tier")
        .expect("tier exists")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("tier is Utf8");
    assert_eq!(name_array.value(0), "Alice");
    assert!(name_array.is_null(1));
    assert!(tier_array.is_null(0));
    assert_eq!(tier_array.value(1), "gold");

    Ok(())
}

#[test]
fn compiled_ndjson_plan_reuses_shape_cache_for_repeated_records() -> Result<(), Box<dyn Error>> {
    let payload_fields: Fields = vec![
        Arc::new(Field::new("score", DataType::Float64, true)),
        Arc::new(Field::new(
            "profile",
            DataType::Struct(vec![Arc::new(Field::new("name", DataType::Utf8, true))].into()),
            true,
        )),
    ]
    .into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int64, true),
        Field::new("org_id", DataType::Int64, true),
        Field::new("payload", DataType::Struct(payload_fields), true),
    ]));
    let plan = CompiledNdjsonPlan::compile(schema.as_ref(), &compaction_options())?;
    let mut batch_builder = NdjsonBatchBuilder::new(schema, plan, 8)?;

    let mut first_line =
        br#"{"event_id":1,"org_id":10,"score":1,"profile":{"name":"Alice"}}"#.to_vec();
    let mut first_buffers = Buffers::default();
    let first_value =
        to_borrowed_value_with_buffers(first_line.as_mut_slice(), &mut first_buffers)?;
    batch_builder.append_record(&first_value)?;

    let mut second_line =
        br#"{"event_id":2,"org_id":20,"score":2,"profile":{"name":"Bob"}}"#.to_vec();
    let mut second_buffers = Buffers::default();
    let second_value =
        to_borrowed_value_with_buffers(second_line.as_mut_slice(), &mut second_buffers)?;
    batch_builder.append_record(&second_value)?;

    assert_eq!(batch_builder.append_shape_cache.len(), 1);
    Ok(())
}

#[test]
fn discover_ndjson_schema_rejects_sort_field_outside_envelope() -> Result<(), Box<dyn Error>> {
    let path = unique_path("discover_sort_field_validation", "ndjson");
    write_ndjson(
        &path,
        &[json!({
            "event_id": 1,
            "org_id": 10,
            "event_time": "2026-04-13T00:00:00Z",
            "score": 1
        })],
    )?;

    let error = discover_ndjson_schema_from_paths(
        &[path.clone()],
        &CompactionOptions {
            sort_field: Some("event_time".to_string()),
            ..compaction_options()
        },
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("sort_field `event_time` must also be listed in envelope_fields"));

    std::fs::remove_file(path)?;
    Ok(())
}

#[test]
fn discover_ndjson_schema_rejects_zero_sort_buffer() -> Result<(), Box<dyn Error>> {
    let path = unique_path("discover_sort_buffer_validation", "ndjson");
    write_ndjson(
        &path,
        &[json!({
            "event_id": 1,
            "org_id": 10,
            "score": 1
        })],
    )?;

    let error = discover_ndjson_schema_from_paths(
        &[path.clone()],
        &CompactionOptions {
            sort_max_rows_in_memory: 0,
            ..compaction_options()
        },
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("sort_max_rows_in_memory must be greater than zero"));

    std::fs::remove_file(path)?;
    Ok(())
}

#[tokio::test]
async fn compact_ndjson_to_parquet_writes_union_payload_schema() -> Result<(), Box<dyn Error>> {
    let left_path = unique_path("compact_left", "ndjson");
    let right_path = unique_path("compact_right", "ndjson");
    let output_path = unique_path("compact_output", "parquet");
    write_ndjson(
        &left_path,
        &[
            json!({
                "event_id": 1,
                "org_id": 10,
                "score": 1,
                "profile": { "name": "Alice" },
                "scores": [1, 2]
            }),
            json!({
                "event_id": 2,
                "org_id": 10,
                "score": 2,
                "profile": { "name": "Bob" },
                "scores": [3, 4]
            }),
        ],
    )?;
    write_ndjson(
        &right_path,
        &[json!({
            "event_id": 3,
            "org_id": 20,
            "score": 4.5,
            "profile": { "tier": "gold" },
            "scores": [1.5],
            "amount": 9
        })],
    )?;

    let report = compact_ndjson_to_parquet(
        &[left_path.clone(), right_path.clone()],
        &output_path,
        &compaction_options(),
    )
    .await?;
    assert_eq!(report.rows, 3);

    let batches = read_parquet_batches(&output_path).await?;
    let merged_batch = batches
        .iter()
        .flat_map(|batch| vec![batch.clone()])
        .next()
        .expect("compaction emitted batches");
    let payload = merged_batch
        .column(merged_batch.schema().index_of("payload")?)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("payload is a struct");
    let payload_fields = match payload.data_type() {
        DataType::Struct(fields) => fields,
        other => panic!("expected payload struct, got {other:?}"),
    };
    let score_index = payload_fields
        .iter()
        .position(|field| field.name() == "score")
        .expect("payload.score exists");
    let amount_index = payload_fields
        .iter()
        .position(|field| field.name() == "amount")
        .expect("payload.amount exists");
    assert_eq!(payload_fields[score_index].data_type(), &DataType::Float64);
    assert_eq!(payload_fields[amount_index].name(), "amount");

    let amount_array = payload
        .column(amount_index)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("payload.amount is Int64");
    assert!(amount_array.is_null(0));
    assert!(amount_array.is_null(1));
    assert_eq!(amount_array.value(2), 9);

    let score_array = payload
        .column(score_index)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("payload.score is Float64");
    assert_eq!(score_array.value(0), 1.0);
    assert_eq!(score_array.value(1), 2.0);
    assert_eq!(score_array.value(2), 4.5);

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(right_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn compact_ndjson_to_parquet_sorts_numeric_field_stably_with_nulls_last()
-> Result<(), Box<dyn Error>> {
    let input_path = unique_path("compact_sort_numeric", "ndjson");
    let output_path = unique_path("compact_sort_numeric_output", "parquet");
    write_ndjson(
        &input_path,
        &[
            json!({ "event_id": 2, "org_id": 10, "score": 20 }),
            json!({ "event_id": null, "org_id": 11, "score": 99 }),
            json!({ "event_id": 1, "org_id": 12, "score": 10 }),
            json!({ "event_id": 1, "org_id": 13, "score": 11 }),
        ],
    )?;

    let report = compact_ndjson_to_parquet(
        &[input_path.clone()],
        &output_path,
        &compaction_options_with_sort("event_id", 16),
    )
    .await?;
    assert_eq!(report.rows, 4);
    assert!(report.sorting_duration <= report.total_duration);

    let batches = read_parquet_batches(&output_path).await?;
    let batch = batches.first().expect("sorted parquet has a batch");
    let event_ids = batch
        .column(batch.schema().index_of("event_id")?)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("event_id is Int64");
    let org_ids = batch
        .column(batch.schema().index_of("org_id")?)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("org_id is Int64");
    assert_eq!(event_ids.value(0), 1);
    assert_eq!(event_ids.value(1), 1);
    assert_eq!(event_ids.value(2), 2);
    assert!(event_ids.is_null(3));
    assert_eq!(org_ids.value(0), 12);
    assert_eq!(org_ids.value(1), 13);

    let _ = tokio::fs::remove_file(input_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn compact_ndjson_to_parquet_sorts_string_field_and_is_file_order_independent()
-> Result<(), Box<dyn Error>> {
    let left_path = unique_path("compact_sort_string_left", "ndjson");
    let right_path = unique_path("compact_sort_string_right", "ndjson");
    let forward_output = unique_path("compact_sort_string_forward", "parquet");
    let reverse_output = unique_path("compact_sort_string_reverse", "parquet");
    write_ndjson(
        &left_path,
        &[
            json!({ "event_time": "2026-04-13T01:00:00Z", "org_id": 10, "score": 1 }),
            json!({ "event_time": "2026-04-13T03:00:00Z", "org_id": 11, "score": 3 }),
        ],
    )?;
    write_ndjson(
        &right_path,
        &[
            json!({ "event_time": "2026-04-13T00:00:00Z", "org_id": 20, "score": 0 }),
            json!({ "event_time": null, "org_id": 21, "score": 9 }),
        ],
    )?;

    let sort_options = CompactionOptions {
        envelope_fields: vec!["event_time".to_string(), "org_id".to_string()],
        sort_field: Some("event_time".to_string()),
        ..compaction_options()
    };

    compact_ndjson_to_parquet(
        &[left_path.clone(), right_path.clone()],
        &forward_output,
        &sort_options,
    )
    .await?;
    compact_ndjson_to_parquet(
        &[right_path.clone(), left_path.clone()],
        &reverse_output,
        &sort_options,
    )
    .await?;

    let forward_batches = read_parquet_batches(&forward_output).await?;
    let reverse_batches = read_parquet_batches(&reverse_output).await?;
    let forward = forward_batches.first().expect("forward output has a batch");
    let reverse = reverse_batches.first().expect("reverse output has a batch");

    let forward_times = forward
        .column(forward.schema().index_of("event_time")?)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("event_time is Utf8");
    let reverse_times = reverse
        .column(reverse.schema().index_of("event_time")?)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("event_time is Utf8");
    let forward_org_ids = forward
        .column(forward.schema().index_of("org_id")?)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("org_id is Int64");
    let reverse_org_ids = reverse
        .column(reverse.schema().index_of("org_id")?)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("org_id is Int64");

    assert_eq!(forward_times.value(0), "2026-04-13T00:00:00Z");
    assert_eq!(forward_times.value(1), "2026-04-13T01:00:00Z");
    assert_eq!(forward_times.value(2), "2026-04-13T03:00:00Z");
    assert!(forward_times.is_null(3));
    assert_eq!(
        forward_org_ids.iter().collect::<Vec<_>>(),
        reverse_org_ids.iter().collect::<Vec<_>>()
    );
    assert_eq!(forward_times.value(0), reverse_times.value(0));
    assert_eq!(forward_times.value(1), reverse_times.value(1));
    assert_eq!(forward_times.value(2), reverse_times.value(2));
    assert!(reverse_times.is_null(3));

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(right_path).await;
    let _ = tokio::fs::remove_file(forward_output).await;
    let _ = tokio::fs::remove_file(reverse_output).await;
    Ok(())
}

#[tokio::test]
async fn compact_ndjson_to_parquet_rejects_unsupported_sort_type() -> Result<(), Box<dyn Error>> {
    let input_path = unique_path("compact_sort_boolean", "ndjson");
    let output_path = unique_path("compact_sort_boolean_output", "parquet");
    write_ndjson(
        &input_path,
        &[json!({ "is_active": true, "org_id": 10, "score": 1 })],
    )?;

    let error = compact_ndjson_to_parquet(
        &[input_path.clone()],
        &output_path,
        &CompactionOptions {
            envelope_fields: vec!["is_active".to_string(), "org_id".to_string()],
            sort_field: Some("is_active".to_string()),
            ..compaction_options()
        },
    )
    .await
    .unwrap_err()
    .to_string();

    assert!(error.contains("unsupported type Boolean"));

    let _ = tokio::fs::remove_file(input_path).await;
    Ok(())
}

#[tokio::test]
async fn compact_ndjson_to_parquet_rejects_sort_jobs_above_memory_cap() -> Result<(), Box<dyn Error>>
{
    let input_path = unique_path("compact_sort_cap", "ndjson");
    let output_path = unique_path("compact_sort_cap_output", "parquet");
    write_ndjson(
        &input_path,
        &[
            json!({ "event_id": 2, "org_id": 10, "score": 20 }),
            json!({ "event_id": 1, "org_id": 11, "score": 10 }),
        ],
    )?;

    let error = compact_ndjson_to_parquet(
        &[input_path.clone()],
        &output_path,
        &compaction_options_with_sort("event_id", 1),
    )
    .await
    .unwrap_err()
    .to_string();

    assert!(error.contains("external spill sort is not implemented"));

    let _ = tokio::fs::remove_file(input_path).await;
    Ok(())
}

#[test]
fn prepared_order_columns_compare_all_supported_key_types() {
    let cases = vec![
        PreparedOrderColumn::Int64(Int64Array::from(vec![Some(1), Some(2), None])),
        PreparedOrderColumn::UInt64(UInt64Array::from(vec![Some(1), Some(2), None])),
        PreparedOrderColumn::Float64(Float64Array::from(vec![Some(1.0), Some(2.0), None])),
        PreparedOrderColumn::Utf8(StringArray::from(vec![Some("a"), Some("b"), None])),
        PreparedOrderColumn::LargeUtf8(LargeStringArray::from(vec![Some("a"), Some("b"), None])),
        PreparedOrderColumn::Date32(Date32Array::from(vec![Some(1), Some(2), None])),
        PreparedOrderColumn::Date64(Date64Array::from(vec![Some(1), Some(2), None])),
        PreparedOrderColumn::TimestampSecond(TimestampSecondArray::from(vec![
            Some(1),
            Some(2),
            None,
        ])),
        PreparedOrderColumn::TimestampMillisecond(TimestampMillisecondArray::from(vec![
            Some(1),
            Some(2),
            None,
        ])),
        PreparedOrderColumn::TimestampMicrosecond(TimestampMicrosecondArray::from(vec![
            Some(1),
            Some(2),
            None,
        ])),
        PreparedOrderColumn::TimestampNanosecond(TimestampNanosecondArray::from(vec![
            Some(1),
            Some(2),
            None,
        ])),
    ];

    for column in cases {
        assert_eq!(column.compare_rows(0, 1), CmpOrdering::Less);
        assert_eq!(column.compare_rows(1, 2), CmpOrdering::Less);
    }
}

#[test]
fn contiguous_run_len_honors_tie_breaks_and_boundaries() {
    let order_array = Int64Array::from(vec![1, 2, 2, 3]);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "event_id",
            DataType::Int64,
            true,
        )])),
        vec![Arc::new(order_array.clone()) as ArrayRef],
    )
    .unwrap();
    let prepared = PreparedOrderBatch {
        batch,
        order_column: PreparedOrderColumn::Int64(order_array),
        row_group_index: 0,
        row_group_batch_index: 0,
        non_null_prefix_len: 4,
    };

    let (_, receiver) = mpsc::channel(1);
    let lower_source = OrderedMergeSource {
        source_index: 0,
        receiver,
        current_batch: Some(prepared.clone()),
        current_row: 0,
    };
    let competitor_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "event_id",
            DataType::Int64,
            true,
        )])),
        vec![Arc::new(Int64Array::from(vec![2])) as ArrayRef],
    )
    .unwrap();
    let (_, receiver) = mpsc::channel(1);
    let competitor = OrderedMergeSource {
        source_index: 1,
        receiver,
        current_batch: Some(PreparedOrderBatch {
            batch: competitor_batch,
            order_column: PreparedOrderColumn::Int64(Int64Array::from(vec![2])),
            row_group_index: 0,
            row_group_batch_index: 0,
            non_null_prefix_len: 1,
        }),
        current_row: 0,
    };
    assert_eq!(lower_source.contiguous_run_len(Some(&competitor), 8), 3);

    let (_, receiver) = mpsc::channel(1);
    let higher_source = OrderedMergeSource {
        source_index: 2,
        receiver,
        current_batch: Some(prepared),
        current_row: 0,
    };
    assert_eq!(higher_source.contiguous_run_len(Some(&competitor), 8), 1);
    assert_eq!(higher_source.contiguous_run_len(Some(&competitor), 1), 1);
}

#[test]
fn validate_prepared_ordered_batch_rejects_descending_and_null_then_value() {
    let descending_array = Int64Array::from(vec![2, 1]);
    let descending_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "event_id",
            DataType::Int64,
            true,
        )])),
        vec![Arc::new(descending_array.clone()) as ArrayRef],
    )
    .unwrap();
    let descending = PreparedOrderBatch {
        batch: descending_batch,
        order_column: PreparedOrderColumn::Int64(descending_array),
        row_group_index: 0,
        row_group_batch_index: 0,
        non_null_prefix_len: 2,
    };
    let mut last_key = None;
    let error = validate_prepared_ordered_batch(
        &descending,
        &mut last_key,
        Path::new("/tmp/descending.parquet"),
        "event_id",
        0,
    )
    .unwrap_err();
    assert!(error.contains("event_id"));

    let null_then_value_array = Int64Array::from(vec![None, Some(1)]);
    let null_then_value_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "event_id",
            DataType::Int64,
            true,
        )])),
        vec![Arc::new(null_then_value_array.clone()) as ArrayRef],
    )
    .unwrap();
    let null_then_value = PreparedOrderBatch {
        batch: null_then_value_batch,
        order_column: PreparedOrderColumn::Int64(null_then_value_array),
        row_group_index: 0,
        row_group_batch_index: 0,
        non_null_prefix_len: 0,
    };
    let mut last_key = None;
    let error = validate_prepared_ordered_batch(
        &null_then_value,
        &mut last_key,
        Path::new("/tmp/null_then_value.parquet"),
        "event_id",
        0,
    )
    .unwrap_err();
    assert!(error.contains("event_id"));
}

#[test]
fn row_group_fast_path_respects_ties_and_nulls_last() {
    let make_source = |source_index: usize, values: Vec<Option<i64>>| {
        let array = Int64Array::from(values.clone());
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "event_id",
                DataType::Int64,
                true,
            )])),
            vec![Arc::new(array.clone()) as ArrayRef],
        )
        .unwrap();
        let (_, receiver) = mpsc::channel(1);
        OrderedMergeSource {
            source_index,
            receiver,
            current_batch: Some(PreparedOrderBatch {
                batch,
                order_column: PreparedOrderColumn::Int64(array),
                row_group_index: 0,
                row_group_batch_index: 0,
                non_null_prefix_len: values.iter().take_while(|value| value.is_some()).count(),
            }),
            current_row: 0,
        }
    };

    let lower_tie = make_source(0, vec![Some(2)]);
    let higher_tie = make_source(1, vec![Some(2)]);
    let null_source = make_source(2, vec![None]);
    let sources = vec![lower_tie, higher_tie, null_source];
    let metadata = vec![
        SourceOrderMetadata {
            row_groups: vec![RowGroupOrderRange {
                row_group_index: 0,
                min: Some(OrderKeyValue::Int64(Some(2))),
                max: Some(OrderKeyValue::Int64(Some(2))),
                null_kind: RowGroupNullKind::NoNulls,
            }],
        },
        SourceOrderMetadata {
            row_groups: vec![RowGroupOrderRange {
                row_group_index: 0,
                min: Some(OrderKeyValue::Int64(Some(2))),
                max: Some(OrderKeyValue::Int64(Some(2))),
                null_kind: RowGroupNullKind::NoNulls,
            }],
        },
        SourceOrderMetadata {
            row_groups: vec![RowGroupOrderRange {
                row_group_index: 0,
                min: None,
                max: None,
                null_kind: RowGroupNullKind::AllNulls,
            }],
        },
    ];

    assert!(row_group_fast_path_safe(0, &sources, &metadata));
    assert!(!row_group_fast_path_safe(1, &sources, &metadata));
    assert!(!row_group_fast_path_safe(2, &sources, &metadata));
}

#[test]
fn parquet_parallelism_resolution_handles_auto_serial_and_caps() {
    let mut options = ParquetMergeExecutionOptions::default();
    assert_eq!(resolve_parquet_parallelism(&options, 0), 0);
    assert_eq!(resolve_parquet_parallelism(&options, 4), 1);

    options.parallelism = 0;
    assert_eq!(
        resolve_parquet_parallelism(&options, 4),
        default_scan_parallelism().min(4)
    );

    options.parallelism = 99;
    assert_eq!(resolve_parquet_parallelism(&options, 3), 3);
}

#[tokio::test]
async fn unordered_payload_parallel_preserves_input_order() -> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, right_schema, right_batch) = sample_payload_inputs();
    let left_path = unique_path("unordered_parallel_left", "parquet");
    let right_path = unique_path("unordered_parallel_right", "parquet");
    let output_path = unique_path("unordered_parallel_preserve", "parquet");

    write_parquet(&left_path, left_schema, left_batch).await?;
    write_parquet(&right_path, right_schema, right_batch).await?;

    let report = merge_payload_parquet_files_with_execution(
        &[left_path.clone(), right_path.clone()],
        &output_path,
        &payload_schema_options(),
        &ParquetMergeExecutionOptions {
            read_batch_size: 1,
            prefetch_batches_per_source: 2,
            parallelism: 0,
            unordered_merge_order: UnorderedMergeOrder::PreserveInputOrder,
            ..ParquetMergeExecutionOptions::default()
        },
    )
    .await?;
    assert_eq!(report.rows, 4);

    let batches = read_parquet_batches(&output_path).await?;
    assert_eq!(
        collect_int32_column(&batches, "event_id"),
        vec![Some(1), Some(2), Some(3), Some(4)]
    );

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(right_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn unordered_payload_parallel_interleaved_preserves_rows_and_schema()
-> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, right_schema, right_batch) = sample_payload_inputs();
    let left_path = unique_path("unordered_interleaved_left", "parquet");
    let right_path = unique_path("unordered_interleaved_right", "parquet");
    let output_path = unique_path("unordered_interleaved_output", "parquet");

    write_parquet(&left_path, left_schema, left_batch).await?;
    write_parquet(&right_path, right_schema, right_batch).await?;

    let report = merge_payload_parquet_files_with_execution(
        &[left_path.clone(), right_path.clone()],
        &output_path,
        &payload_schema_options(),
        &ParquetMergeExecutionOptions {
            read_batch_size: 1,
            prefetch_batches_per_source: 2,
            parallelism: 0,
            unordered_merge_order: UnorderedMergeOrder::AllowInterleaved,
            ..ParquetMergeExecutionOptions::default()
        },
    )
    .await?;
    assert_eq!(report.rows, 4);

    let batches = read_parquet_batches(&output_path).await?;
    let merged_schema = batches.first().expect("output batch exists").schema();
    assert_eq!(collect_int32_column(&batches, "event_id").len(), 4);
    assert!(merged_schema.field_with_name("payload").is_ok());

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(right_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn top_level_parallel_execution_preserves_input_order() -> Result<(), Box<dyn Error>> {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int32, false),
        Field::new("score", DataType::Int32, true),
    ]));
    let left_batch = RecordBatch::try_new(
        left_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(10), Some(20)])) as ArrayRef,
        ],
    )?;
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("score", DataType::Float64, true),
        Field::new("event_id", DataType::Int32, false),
    ]));
    let right_batch = RecordBatch::try_new(
        right_schema.clone(),
        vec![
            Arc::new(Float64Array::from(vec![Some(30.0), Some(40.0)])) as ArrayRef,
            Arc::new(Int32Array::from(vec![3, 4])) as ArrayRef,
        ],
    )?;
    let left_path = unique_path("top_level_parallel_left", "parquet");
    let right_path = unique_path("top_level_parallel_right", "parquet");
    let output_path = unique_path("top_level_parallel_output", "parquet");

    write_parquet(&left_path, left_schema, left_batch).await?;
    write_parquet(&right_path, right_schema, right_batch).await?;

    let report = merge_top_level_parquet_files_with_execution(
        &[left_path.clone(), right_path.clone()],
        &output_path,
        &TopLevelMergeOptions::default(),
        &ParquetMergeExecutionOptions {
            read_batch_size: 1,
            prefetch_batches_per_source: 2,
            parallelism: 0,
            unordered_merge_order: UnorderedMergeOrder::PreserveInputOrder,
            ..ParquetMergeExecutionOptions::default()
        },
    )
    .await?;
    assert_eq!(report.rows, 4);

    let batches = read_parquet_batches(&output_path).await?;
    assert_eq!(
        collect_int32_column(&batches, "event_id"),
        vec![Some(1), Some(2), Some(3), Some(4)]
    );
    assert_eq!(
        batches[0].schema().field_with_name("score")?.data_type(),
        &DataType::Float64
    );

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(right_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[test]
fn parquet_compression_options_validate_and_map() -> Result<(), Box<dyn Error>> {
    let defaults = ParquetMergeExecutionOptions::default();
    assert_eq!(
        defaults.writer_compression,
        ParquetCompression::Uncompressed
    );
    assert!(defaults.writer_dictionary_enabled);

    let snappy = ParquetMergeExecutionOptions {
        writer_compression: ParquetCompression::Snappy,
        ..ParquetMergeExecutionOptions::default()
    };
    let properties = parquet_writer_properties(&snappy).unwrap();
    assert_eq!(
        properties.compression(&parquet::schema::types::ColumnPath::from("x")),
        Compression::SNAPPY
    );

    let invalid = ParquetMergeExecutionOptions {
        writer_compression: ParquetCompression::Zstd { level: 0 },
        ..ParquetMergeExecutionOptions::default()
    };
    assert!(validate_parquet_merge_execution_options(&invalid).is_err());
    Ok(())
}

#[tokio::test]
async fn parallel_writer_outputs_requested_compression_and_partial_row_group()
-> Result<(), Box<dyn Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let left_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])) as ArrayRef,
        ],
    )?;
    let right_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("d"), Some("e")])) as ArrayRef,
        ],
    )?;
    let left_path = unique_path("parallel_compression_left", "parquet");
    let right_path = unique_path("parallel_compression_right", "parquet");
    let output_path = unique_path("parallel_compression_output", "parquet");

    write_parquet(&left_path, schema.clone(), left_batch).await?;
    write_parquet(&right_path, schema, right_batch).await?;

    let report = merge_top_level_parquet_files_with_execution(
        &[left_path.clone(), right_path.clone()],
        &output_path,
        &TopLevelMergeOptions::default(),
        &ParquetMergeExecutionOptions {
            read_batch_size: 1,
            output_batch_rows: 1,
            output_row_group_rows: 2,
            prefetch_batches_per_source: 2,
            parallelism: 2,
            writer_compression: ParquetCompression::Snappy,
            ..ParquetMergeExecutionOptions::default()
        },
    )
    .await?;
    assert_eq!(report.rows, 5);
    assert_eq!(parquet_row_group_count(&output_path)?, 3);
    assert!(report.writer_encode_duration > Duration::default());
    assert!(report.writer_sink_duration > Duration::default());
    assert!(report.writer_close_duration > Duration::default());

    let batches = read_parquet_batches(&output_path).await?;
    assert_eq!(
        collect_int32_column(&batches, "event_id"),
        vec![Some(1), Some(2), Some(3), Some(4), Some(5)]
    );
    let compressions = parquet_compressions(&output_path)?;
    assert!(!compressions.is_empty());
    assert!(
        compressions
            .iter()
            .all(|codec| *codec == Compression::SNAPPY)
    );

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(right_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn parallel_writer_supports_lz4_raw_and_zstd() -> Result<(), Box<dyn Error>> {
    for (writer_compression, expected) in [
        (ParquetCompression::Lz4Raw, Compression::LZ4_RAW),
        (
            ParquetCompression::Zstd { level: 1 },
            Compression::ZSTD(ZstdLevel::try_new(1)?),
        ),
    ] {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "event_id",
            DataType::Int32,
            false,
        )]));
        let left_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
        )?;
        let right_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![3, 4])) as ArrayRef],
        )?;
        let left_path = unique_path("parallel_codec_left", "parquet");
        let right_path = unique_path("parallel_codec_right", "parquet");
        let output_path = unique_path("parallel_codec_output", "parquet");

        write_parquet(&left_path, schema.clone(), left_batch).await?;
        write_parquet(&right_path, schema, right_batch).await?;

        let report = merge_top_level_parquet_files_with_execution(
            &[left_path.clone(), right_path.clone()],
            &output_path,
            &TopLevelMergeOptions::default(),
            &ParquetMergeExecutionOptions {
                read_batch_size: 1,
                output_batch_rows: 1,
                output_row_group_rows: 2,
                parallelism: 2,
                writer_compression,
                ..ParquetMergeExecutionOptions::default()
            },
        )
        .await?;
        assert_eq!(report.rows, 4);
        assert!(
            parquet_compressions(&output_path)?
                .iter()
                .all(|codec| *codec == expected)
        );

        let _ = tokio::fs::remove_file(left_path).await;
        let _ = tokio::fs::remove_file(right_path).await;
        let _ = tokio::fs::remove_file(output_path).await;
    }
    Ok(())
}

#[tokio::test]
async fn ordered_merge_respects_parallelism_settings() -> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, right_schema, right_batch) = sample_payload_inputs();
    let (left_schema, left_batch) =
        promote_payload_envelope_to_int64(left_schema, left_batch, &[1, 3], &[10, 20]);
    let (right_schema, right_batch) =
        promote_payload_envelope_to_int64(right_schema, right_batch, &[2, 4], &[30, 40]);
    let left_path = unique_path("ordered_parallel_left", "parquet");
    let right_path = unique_path("ordered_parallel_right", "parquet");
    write_parquet(&left_path, left_schema, left_batch).await?;
    write_parquet(&right_path, right_schema, right_batch).await?;

    for parallelism in [2, 0] {
        let output_path = unique_path(&format!("ordered_parallel_output_{parallelism}"), "parquet");
        let report = merge_payload_parquet_files_with_execution(
            &[left_path.clone(), right_path.clone()],
            &output_path,
            &payload_schema_options(),
            &ParquetMergeExecutionOptions {
                parallelism,
                ..ordered_execution_options("event_id")
            },
        )
        .await?;
        assert_eq!(report.rows, 4);
        assert_eq!(
            collect_int64_column(&read_parquet_batches(&output_path).await?, "event_id"),
            vec![Some(1), Some(2), Some(3), Some(4)]
        );
        let _ = tokio::fs::remove_file(output_path).await;
    }

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(right_path).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_typed_drivers_cover_supported_key_types() -> Result<(), Box<dyn Error>> {
    let cases: Vec<(&str, DataType, ArrayRef, ArrayRef, Vec<Option<String>>)> = vec![
        (
            "int64",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![1, 3])) as ArrayRef,
            Arc::new(Int64Array::from(vec![2, 4])) as ArrayRef,
            ["1", "2", "3", "4"]
                .into_iter()
                .map(|value| Some(value.to_string()))
                .collect(),
        ),
        (
            "uint64",
            DataType::UInt64,
            Arc::new(UInt64Array::from(vec![1_u64, 3])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![2_u64, 4])) as ArrayRef,
            ["1", "2", "3", "4"]
                .into_iter()
                .map(|value| Some(value.to_string()))
                .collect(),
        ),
        (
            "float64",
            DataType::Float64,
            Arc::new(Float64Array::from(vec![1.0, 3.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![2.0, 4.0])) as ArrayRef,
            ["1", "2", "3", "4"]
                .into_iter()
                .map(|value| Some(value.to_string()))
                .collect(),
        ),
        (
            "utf8",
            DataType::Utf8,
            Arc::new(StringArray::from(vec![Some("a"), Some("c")])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("b"), Some("d")])) as ArrayRef,
            ["a", "b", "c", "d"]
                .into_iter()
                .map(|value| Some(value.to_string()))
                .collect(),
        ),
        (
            "large_utf8",
            DataType::LargeUtf8,
            Arc::new(LargeStringArray::from(vec![Some("a"), Some("c")])) as ArrayRef,
            Arc::new(LargeStringArray::from(vec![Some("b"), Some("d")])) as ArrayRef,
            ["a", "b", "c", "d"]
                .into_iter()
                .map(|value| Some(value.to_string()))
                .collect(),
        ),
        (
            "date32",
            DataType::Date32,
            Arc::new(Date32Array::from(vec![1, 3])) as ArrayRef,
            Arc::new(Date32Array::from(vec![2, 4])) as ArrayRef,
            ["1", "2", "3", "4"]
                .into_iter()
                .map(|value| Some(value.to_string()))
                .collect(),
        ),
        (
            "date64",
            DataType::Date64,
            Arc::new(Date64Array::from(vec![1_i64, 3])) as ArrayRef,
            Arc::new(Date64Array::from(vec![2_i64, 4])) as ArrayRef,
            ["1", "2", "3", "4"]
                .into_iter()
                .map(|value| Some(value.to_string()))
                .collect(),
        ),
        (
            "timestamp_second",
            DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
            Arc::new(TimestampSecondArray::from(vec![1_i64, 3])) as ArrayRef,
            Arc::new(TimestampSecondArray::from(vec![2_i64, 4])) as ArrayRef,
            ["1", "2", "3", "4"]
                .into_iter()
                .map(|value| Some(value.to_string()))
                .collect(),
        ),
        (
            "timestamp_millisecond",
            DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![1_i64, 3])) as ArrayRef,
            Arc::new(TimestampMillisecondArray::from(vec![2_i64, 4])) as ArrayRef,
            ["1", "2", "3", "4"]
                .into_iter()
                .map(|value| Some(value.to_string()))
                .collect(),
        ),
        (
            "timestamp_microsecond",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            Arc::new(TimestampMicrosecondArray::from(vec![1_i64, 3])) as ArrayRef,
            Arc::new(TimestampMicrosecondArray::from(vec![2_i64, 4])) as ArrayRef,
            ["1", "2", "3", "4"]
                .into_iter()
                .map(|value| Some(value.to_string()))
                .collect(),
        ),
        (
            "timestamp_nanosecond",
            DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
            Arc::new(TimestampNanosecondArray::from(vec![1_i64, 3])) as ArrayRef,
            Arc::new(TimestampNanosecondArray::from(vec![2_i64, 4])) as ArrayRef,
            ["1", "2", "3", "4"]
                .into_iter()
                .map(|value| Some(value.to_string()))
                .collect(),
        ),
    ];

    let payload_fields: Fields = vec![Arc::new(Field::new("score", DataType::Int32, true))].into();

    for (name, data_type, left_key, right_key, expected) in cases {
        let schema = Arc::new(Schema::new(vec![
            Field::new("order_key", data_type, false),
            Field::new("org_id", DataType::Int64, false),
            Field::new("payload", DataType::Struct(payload_fields.clone()), true),
        ]));
        let make_batch =
            |key: ArrayRef, org_ids: Vec<i64>| -> Result<RecordBatch, Box<dyn Error>> {
                let payload = Arc::new(StructArray::new(
                    payload_fields.clone(),
                    vec![Arc::new(Int32Array::from(vec![Some(10), Some(20)])) as ArrayRef],
                    None,
                )) as ArrayRef;
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        key,
                        Arc::new(Int64Array::from(org_ids)) as ArrayRef,
                        payload,
                    ],
                )
                .map_err(|error| io_error(error.to_string()).into())
            };

        let left_path = unique_path(&format!("ordered_typed_{name}_left"), "parquet");
        let right_path = unique_path(&format!("ordered_typed_{name}_right"), "parquet");
        let output_path = unique_path(&format!("ordered_typed_{name}_output"), "parquet");
        write_parquet(
            &left_path,
            schema.clone(),
            make_batch(left_key, vec![10, 30])?,
        )
        .await?;
        write_parquet(
            &right_path,
            schema.clone(),
            make_batch(right_key, vec![20, 40])?,
        )
        .await?;

        let report = merge_payload_parquet_files_with_execution(
            &[left_path.clone(), right_path.clone()],
            &output_path,
            &payload_schema_options(),
            &ParquetMergeExecutionOptions {
                parallelism: 2,
                output_batch_rows: 3,
                output_row_group_rows: 3,
                ..ordered_execution_options("order_key")
            },
        )
        .await?;

        assert_eq!(report.rows, 4, "{name}");
        assert_eq!(
            collect_order_key_column(&read_parquet_batches(&output_path).await?, "order_key"),
            expected,
            "{name}"
        );

        let _ = tokio::fs::remove_file(left_path).await;
        let _ = tokio::fs::remove_file(right_path).await;
        let _ = tokio::fs::remove_file(output_path).await;
    }

    Ok(())
}

fn collect_int64_column(batches: &[RecordBatch], field_name: &str) -> Vec<Option<i64>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(batch.schema().index_of(field_name).unwrap())
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}

fn collect_int32_column(batches: &[RecordBatch], field_name: &str) -> Vec<Option<i32>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(batch.schema().index_of(field_name).unwrap())
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}

fn collect_string_column(batches: &[RecordBatch], field_name: &str) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(batch.schema().index_of(field_name).unwrap())
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|value| value.map(str::to_string))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn collect_order_key_column(batches: &[RecordBatch], field_name: &str) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|batch| {
            let column = batch.column(batch.schema().index_of(field_name).unwrap());
            match column.data_type() {
                DataType::Int64 => column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(|value| value.to_string()))
                    .collect::<Vec<_>>(),
                DataType::UInt64 => column
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(|value| value.to_string()))
                    .collect::<Vec<_>>(),
                DataType::Float64 => column
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(|value| value.to_string()))
                    .collect::<Vec<_>>(),
                DataType::Utf8 => column
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(str::to_string))
                    .collect::<Vec<_>>(),
                DataType::LargeUtf8 => column
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(str::to_string))
                    .collect::<Vec<_>>(),
                DataType::Date32 => column
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(|value| value.to_string()))
                    .collect::<Vec<_>>(),
                DataType::Date64 => column
                    .as_any()
                    .downcast_ref::<Date64Array>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(|value| value.to_string()))
                    .collect::<Vec<_>>(),
                DataType::Timestamp(arrow_schema::TimeUnit::Second, _) => column
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(|value| value.to_string()))
                    .collect::<Vec<_>>(),
                DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => column
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(|value| value.to_string()))
                    .collect::<Vec<_>>(),
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => column
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(|value| value.to_string()))
                    .collect::<Vec<_>>(),
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => column
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(|value| value.to_string()))
                    .collect::<Vec<_>>(),
                other => panic!("unsupported ordered test column type: {other:?}"),
            }
        })
        .collect()
}

fn concat_batch_slices(
    schema: SchemaRef,
    slices: &[(RecordBatch, usize, usize)],
) -> Result<RecordBatch, Box<dyn Error>> {
    let pending = slices
        .iter()
        .map(|(batch, start, len)| batch.slice(*start, *len))
        .collect::<Vec<_>>();
    concat_batches(&schema, &pending).map_err(|error| io_error(error.to_string()).into())
}

#[tokio::test]
async fn ordered_output_pipeline_preserves_out_of_order_completion_sequence()
-> Result<(), Box<dyn Error>> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "event_id",
        DataType::Int64,
        false,
    )]));
    let first = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![1])) as ArrayRef],
    )?;
    let second = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![2])) as ArrayRef],
    )?;
    let output_path = unique_path("ordered_pipeline_out_of_order", "parquet");
    let options = ParquetMergeExecutionOptions {
        output_batch_rows: 1,
        output_row_group_rows: 1,
        ..ParquetMergeExecutionOptions::default()
    };
    let mut writer = create_parquet_output_writer(&output_path, schema.clone(), &options, 1, 2)
        .await
        .map_err(io_error)?;
    let mut pipeline = OrderedOutputPipeline::new(2);
    let mut stats = ParquetMergeRunStats::default();

    pipeline
        .record_completed(
            OrderedCompletedOutput {
                sequence: 1,
                item: OrderedCompletedOutputItem::Batch {
                    flush: OrderedFlush {
                        batch: second,
                        mode: OrderedFlushMode::Direct,
                    },
                    count_materialization: false,
                },
                materialization_duration: Duration::default(),
            },
            &mut writer,
            &mut stats,
        )
        .await
        .map_err(io_error)?;
    pipeline
        .record_completed(
            OrderedCompletedOutput {
                sequence: 0,
                item: OrderedCompletedOutputItem::Batch {
                    flush: OrderedFlush {
                        batch: first,
                        mode: OrderedFlushMode::Direct,
                    },
                    count_materialization: false,
                },
                materialization_duration: Duration::default(),
            },
            &mut writer,
            &mut stats,
        )
        .await
        .map_err(io_error)?;
    pipeline
        .finish(&mut writer, &mut stats)
        .await
        .map_err(io_error)?;
    writer.finish().await.map_err(io_error)?;

    assert_eq!(
        collect_int64_column(&read_parquet_batches(&output_path).await?, "event_id"),
        vec![Some(1), Some(2)]
    );
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[test]
fn ordered_output_accumulator_matches_concat_for_primitive_fragments() -> Result<(), Box<dyn Error>>
{
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int64, false),
        Field::new("org_id", DataType::Int64, false),
    ]));
    let first = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef,
        ],
    )?;
    let second = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![4, 5, 6])) as ArrayRef,
            Arc::new(Int64Array::from(vec![40, 50, 60])) as ArrayRef,
        ],
    )?;

    let mut accumulator = OrderedOutputAccumulator::new(schema.clone());
    accumulator.append_range(&first, 0, 2).map_err(io_error)?;
    accumulator.append_range(&second, 1, 2).map_err(io_error)?;
    let actual = accumulator
        .flush()
        .map_err(io_error)?
        .expect("flush emits batch");
    let expected = concat_batch_slices(schema, &[(first, 0, 2), (second, 1, 2)])?;

    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn ordered_output_accumulator_matches_concat_for_struct_fragments() -> Result<(), Box<dyn Error>> {
    let profile_fields: Fields = vec![
        Arc::new(Field::new("name", DataType::Utf8, true)),
        Arc::new(Field::new("tier", DataType::Utf8, true)),
    ]
    .into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int64, false),
        Field::new("profile", DataType::Struct(profile_fields.clone()), true),
    ]));

    let first_profile = Arc::new(StructArray::new(
        profile_fields.clone(),
        vec![
            Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob")])) as ArrayRef,
            Arc::new(StringArray::from(vec![None, Some("gold")])) as ArrayRef,
        ],
        None,
    )) as ArrayRef;
    let second_profile = Arc::new(StructArray::new(
        profile_fields,
        vec![
            Arc::new(StringArray::from(vec![Some("Cara"), None])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("silver"), Some("bronze")])) as ArrayRef,
        ],
        None,
    )) as ArrayRef;

    let first = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            first_profile,
        ],
    )?;
    let second = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![3, 4])) as ArrayRef,
            second_profile,
        ],
    )?;

    let mut accumulator = OrderedOutputAccumulator::new(schema.clone());
    accumulator.append_range(&first, 1, 1).map_err(io_error)?;
    accumulator.append_range(&second, 0, 2).map_err(io_error)?;
    let actual = accumulator
        .flush()
        .map_err(io_error)?
        .expect("flush emits batch");
    let expected = concat_batch_slices(schema, &[(first, 1, 1), (second, 0, 2)])?;

    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn ordered_output_accumulator_matches_concat_for_widened_lists() -> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, right_schema, right_batch) = sample_payload_inputs();
    let mut plan = build_compiled_payload_plan(
        [left_schema.as_ref(), right_schema.as_ref()],
        &payload_schema_options(),
    )
    .map_err(io_error)?;
    let left = plan.adapt_batch(&left_batch)?;
    let right = plan.adapt_batch(&right_batch)?;

    let mut accumulator = OrderedOutputAccumulator::new(plan.output_schema.clone());
    accumulator.append_range(&left, 0, 1).map_err(io_error)?;
    accumulator.append_range(&right, 0, 2).map_err(io_error)?;
    let actual = accumulator
        .flush()
        .map_err(io_error)?
        .expect("flush emits batch");
    let expected = concat_batch_slices(
        plan.output_schema.clone(),
        &[(left.clone(), 0, 1), (right.clone(), 0, 2)],
    )?;

    assert_eq!(actual, expected);

    let payload = actual
        .column(actual.schema().index_of("payload")?)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("payload is struct");
    let scores = payload
        .column_by_name("scores")
        .expect("scores exists")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("scores is list");
    let values = scores
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("widened list values are Float64");
    assert_eq!(values.value(0), 1.0);
    assert_eq!(values.value(1), 2.0);
    assert_eq!(values.value(2), 1.5);

    Ok(())
}

#[test]
fn ordered_output_accumulator_uses_interleave_for_dense_fragments() -> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, right_schema, right_batch) = sample_payload_inputs();
    let mut plan = build_compiled_payload_plan(
        [left_schema.as_ref(), right_schema.as_ref()],
        &payload_schema_options(),
    )
    .map_err(io_error)?;
    let left = plan.adapt_batch(&left_batch)?;
    let right = plan.adapt_batch(&right_batch)?;
    let mut accumulator = OrderedOutputAccumulator::new(plan.output_schema.clone());
    let mut expected_slices = Vec::new();

    for index in 0..10 {
        let (batch, row) = if index % 2 == 0 {
            (left.clone(), 0)
        } else {
            (right.clone(), 1)
        };
        accumulator.append_range(&batch, row, 1).map_err(io_error)?;
        expected_slices.push((batch, row, 1));
    }

    let flushed = accumulator
        .flush_with_mode()
        .map_err(io_error)?
        .expect("flush emits batch");
    let expected = concat_batch_slices(plan.output_schema.clone(), &expected_slices)?;

    assert_eq!(flushed.mode, OrderedFlushMode::Interleave);
    assert_eq!(flushed.batch, expected);
    Ok(())
}

#[test]
fn ordered_output_accumulator_keeps_long_fragments_on_concat_path() -> Result<(), Box<dyn Error>> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "event_id",
        DataType::Int64,
        false,
    )]));
    let first = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..2_000)) as ArrayRef],
    )?;
    let second = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(2_000..4_000)) as ArrayRef],
    )?;
    let mut accumulator = OrderedOutputAccumulator::new(schema.clone());
    accumulator
        .append_range(&first, 0, 1_500)
        .map_err(io_error)?;
    accumulator
        .append_range(&second, 250, 1_500)
        .map_err(io_error)?;

    let flushed = accumulator
        .flush_with_mode()
        .map_err(io_error)?
        .expect("flush emits batch");
    let expected = concat_batch_slices(schema, &[(first, 0, 1_500), (second, 250, 1_500)])?;

    assert_eq!(flushed.mode, OrderedFlushMode::Concat);
    assert_eq!(flushed.batch, expected);
    Ok(())
}

#[tokio::test]
async fn ordered_merge_uses_stats_fast_path_for_non_overlapping_row_groups()
-> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, _, _) = sample_payload_inputs();
    let (first_schema, first_batch) = promote_payload_envelope_to_int64(
        left_schema.clone(),
        left_batch.clone(),
        &[1, 2],
        &[10, 20],
    );
    let (second_schema, second_batch) =
        promote_payload_envelope_to_int64(left_schema, left_batch, &[3, 4], &[30, 40]);
    let writer_properties = WriterProperties::builder()
        .set_max_row_group_size(1)
        .build();
    let first_path = unique_path("ordered_fast_path_left", "parquet");
    let second_path = unique_path("ordered_fast_path_right", "parquet");
    let output_path = unique_path("ordered_fast_path_output", "parquet");

    write_parquet_with_properties(
        &first_path,
        first_schema,
        first_batch,
        Some(writer_properties.clone()),
    )
    .await?;
    write_parquet_with_properties(
        &second_path,
        second_schema,
        second_batch,
        Some(writer_properties),
    )
    .await?;

    let report = merge_payload_parquet_files_with_execution(
        &[first_path.clone(), second_path.clone()],
        &output_path,
        &payload_schema_options(),
        &ordered_execution_options("event_id"),
    )
    .await?;
    assert!(report.fast_path_row_groups >= 2);
    assert!(report.fast_path_batches >= 2);

    let batches = read_parquet_batches(&output_path).await?;
    assert_eq!(
        collect_int64_column(&batches, "event_id"),
        vec![Some(1), Some(2), Some(3), Some(4)]
    );

    let _ = tokio::fs::remove_file(first_path).await;
    let _ = tokio::fs::remove_file(second_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_copies_non_overlapping_exact_schema_row_groups() -> Result<(), Box<dyn Error>>
{
    let payload_fields: Fields = vec![Arc::new(Field::new("score", DataType::Int32, true))].into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int64, false),
        Field::new("org_id", DataType::Int64, false),
        Field::new("payload", DataType::Struct(payload_fields.clone()), true),
    ]));
    let make_batch =
        |event_ids: Vec<i64>, org_ids: Vec<i64>| -> Result<RecordBatch, Box<dyn Error>> {
            let payload = Arc::new(StructArray::new(
                payload_fields.clone(),
                vec![Arc::new(Int32Array::from(
                    event_ids
                        .iter()
                        .map(|value| Some(*value as i32))
                        .collect::<Vec<_>>(),
                )) as ArrayRef],
                None,
            )) as ArrayRef;
            Ok(RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(event_ids)) as ArrayRef,
                    Arc::new(Int64Array::from(org_ids)) as ArrayRef,
                    payload,
                ],
            )?)
        };
    let writer_properties = WriterProperties::builder()
        .set_max_row_group_size(2)
        .set_compression(Compression::SNAPPY)
        .build();
    let first_path = unique_path("ordered_copy_left", "parquet");
    let second_path = unique_path("ordered_copy_right", "parquet");
    let output_path = unique_path("ordered_copy_output", "parquet");
    write_parquet_with_properties(
        &first_path,
        schema.clone(),
        make_batch(vec![1, 2], vec![10, 20])?,
        Some(writer_properties.clone()),
    )
    .await?;
    write_parquet_with_properties(
        &second_path,
        schema.clone(),
        make_batch(vec![3, 4], vec![30, 40])?,
        Some(writer_properties),
    )
    .await?;

    let report = merge_payload_parquet_files_with_execution(
        &[first_path.clone(), second_path.clone()],
        &output_path,
        &payload_schema_options(),
        &ParquetMergeExecutionOptions {
            parallelism: 2,
            writer_compression: ParquetCompression::Snappy,
            output_row_group_rows: 2,
            ..ordered_execution_options("event_id")
        },
    )
    .await?;

    assert_eq!(report.copy_candidate_row_groups, 2);
    assert_eq!(report.copied_row_groups, 2);
    assert_eq!(report.copied_rows, 4);
    assert!(report.copied_compressed_bytes > 0);
    assert_eq!(
        collect_int64_column(&read_parquet_batches(&output_path).await?, "event_id"),
        vec![Some(1), Some(2), Some(3), Some(4)]
    );

    let _ = tokio::fs::remove_file(first_path).await;
    let _ = tokio::fs::remove_file(second_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_copy_falls_back_on_compression_mismatch() -> Result<(), Box<dyn Error>> {
    let payload_fields: Fields = vec![Arc::new(Field::new("score", DataType::Int32, true))].into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int64, false),
        Field::new("org_id", DataType::Int64, false),
        Field::new("payload", DataType::Struct(payload_fields.clone()), true),
    ]));
    let payload = Arc::new(StructArray::new(
        payload_fields,
        vec![Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef],
        None,
    )) as ArrayRef;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20])) as ArrayRef,
            payload,
        ],
    )?;
    let input_path = unique_path("ordered_copy_mismatch_input", "parquet");
    let output_path = unique_path("ordered_copy_mismatch_output", "parquet");
    write_parquet_with_properties(
        &input_path,
        schema,
        batch,
        Some(
            WriterProperties::builder()
                .set_max_row_group_size(2)
                .set_compression(Compression::UNCOMPRESSED)
                .build(),
        ),
    )
    .await?;

    let report = merge_payload_parquet_files_with_execution(
        &[input_path.clone()],
        &output_path,
        &payload_schema_options(),
        &ParquetMergeExecutionOptions {
            parallelism: 2,
            writer_compression: ParquetCompression::Snappy,
            output_row_group_rows: 2,
            ..ordered_execution_options("event_id")
        },
    )
    .await?;

    assert_eq!(report.copy_candidate_row_groups, 0);
    assert_eq!(report.copied_row_groups, 0);
    assert_eq!(
        collect_int64_column(&read_parquet_batches(&output_path).await?, "event_id"),
        vec![Some(1), Some(2)]
    );

    let _ = tokio::fs::remove_file(input_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_supports_string_fast_path() -> Result<(), Box<dyn Error>> {
    let payload_fields: Fields = vec![Arc::new(Field::new("score", DataType::Int32, true))].into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_time", DataType::Utf8, false),
        Field::new("org_id", DataType::Int64, false),
        Field::new("payload", DataType::Struct(payload_fields.clone()), true),
    ]));

    let left_payload = Arc::new(StructArray::new(
        payload_fields.clone(),
        vec![Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef],
        None,
    )) as ArrayRef;
    let right_payload = Arc::new(StructArray::new(
        payload_fields.clone(),
        vec![Arc::new(Int32Array::from(vec![Some(3), Some(4)])) as ArrayRef],
        None,
    )) as ArrayRef;
    let left_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![
                Some("2026-04-13T01:00:00Z"),
                Some("2026-04-13T02:00:00Z"),
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20])) as ArrayRef,
            left_payload,
        ],
    )?;
    let right_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![
                Some("2026-04-13T03:00:00Z"),
                Some("2026-04-13T04:00:00Z"),
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![30, 40])) as ArrayRef,
            right_payload,
        ],
    )?;

    let writer_properties = WriterProperties::builder()
        .set_max_row_group_size(1)
        .build();
    let left_path = unique_path("ordered_string_fast_path_left", "parquet");
    let right_path = unique_path("ordered_string_fast_path_right", "parquet");
    let output_path = unique_path("ordered_string_fast_path_output", "parquet");
    write_parquet_with_properties(
        &left_path,
        schema.clone(),
        left_batch,
        Some(writer_properties.clone()),
    )
    .await?;
    write_parquet_with_properties(&right_path, schema, right_batch, Some(writer_properties))
        .await?;

    let report = merge_payload_parquet_files_with_execution(
        &[left_path.clone(), right_path.clone()],
        &output_path,
        &PayloadMergeOptions::default(),
        &ordered_execution_options("event_time"),
    )
    .await?;
    assert!(report.fast_path_row_groups >= 2);

    let batches = read_parquet_batches(&output_path).await?;
    assert_eq!(
        collect_string_column(&batches, "event_time"),
        vec![
            Some("2026-04-13T01:00:00Z".to_string()),
            Some("2026-04-13T02:00:00Z".to_string()),
            Some("2026-04-13T03:00:00Z".to_string()),
            Some("2026-04-13T04:00:00Z".to_string()),
        ]
    );

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(right_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_falls_back_when_statistics_are_missing() -> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, _, _) = sample_payload_inputs();
    let (first_schema, first_batch) = promote_payload_envelope_to_int64(
        left_schema.clone(),
        left_batch.clone(),
        &[1, 2],
        &[10, 20],
    );
    let (second_schema, second_batch) =
        promote_payload_envelope_to_int64(left_schema, left_batch, &[3, 4], &[30, 40]);
    let writer_properties = WriterProperties::builder()
        .set_max_row_group_size(1)
        .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
        .build();
    let first_path = unique_path("ordered_missing_stats_left", "parquet");
    let second_path = unique_path("ordered_missing_stats_right", "parquet");
    let enabled_output = unique_path("ordered_missing_stats_enabled", "parquet");
    let disabled_output = unique_path("ordered_missing_stats_disabled", "parquet");

    write_parquet_with_properties(
        &first_path,
        first_schema,
        first_batch,
        Some(writer_properties.clone()),
    )
    .await?;
    write_parquet_with_properties(
        &second_path,
        second_schema,
        second_batch,
        Some(writer_properties),
    )
    .await?;

    let enabled_report = merge_payload_parquet_files_with_execution(
        &[first_path.clone(), second_path.clone()],
        &enabled_output,
        &payload_schema_options(),
        &ordered_execution_options("event_id"),
    )
    .await?;
    let disabled_report = merge_payload_parquet_files_with_execution(
        &[first_path.clone(), second_path.clone()],
        &disabled_output,
        &payload_schema_options(),
        &ParquetMergeExecutionOptions {
            stats_fast_path: false,
            ..ordered_execution_options("event_id")
        },
    )
    .await?;
    assert_eq!(enabled_report.fast_path_row_groups, 0);
    assert!(enabled_report.fallback_batches > 0);

    let enabled_batches = read_parquet_batches(&enabled_output).await?;
    let disabled_batches = read_parquet_batches(&disabled_output).await?;
    assert_eq!(
        collect_int64_column(&enabled_batches, "event_id"),
        collect_int64_column(&disabled_batches, "event_id")
    );

    let _ = tokio::fs::remove_file(first_path).await;
    let _ = tokio::fs::remove_file(second_path).await;
    let _ = tokio::fs::remove_file(enabled_output).await;
    let _ = tokio::fs::remove_file(disabled_output).await;
    let _ = disabled_report;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_fast_path_toggle_preserves_output() -> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, _, _) = sample_payload_inputs();
    let (first_schema, first_batch) = promote_payload_envelope_to_int64(
        left_schema.clone(),
        left_batch.clone(),
        &[1, 2],
        &[10, 20],
    );
    let (second_schema, second_batch) =
        promote_payload_envelope_to_int64(left_schema, left_batch, &[3, 4], &[30, 40]);
    let writer_properties = WriterProperties::builder()
        .set_max_row_group_size(1)
        .build();
    let first_path = unique_path("ordered_toggle_left", "parquet");
    let second_path = unique_path("ordered_toggle_right", "parquet");
    let enabled_output = unique_path("ordered_toggle_enabled", "parquet");
    let disabled_output = unique_path("ordered_toggle_disabled", "parquet");

    write_parquet_with_properties(
        &first_path,
        first_schema,
        first_batch,
        Some(writer_properties.clone()),
    )
    .await?;
    write_parquet_with_properties(
        &second_path,
        second_schema,
        second_batch,
        Some(writer_properties),
    )
    .await?;

    let enabled_report = merge_payload_parquet_files_with_execution(
        &[first_path.clone(), second_path.clone()],
        &enabled_output,
        &payload_schema_options(),
        &ordered_execution_options("event_id"),
    )
    .await?;
    let disabled_report = merge_payload_parquet_files_with_execution(
        &[first_path.clone(), second_path.clone()],
        &disabled_output,
        &payload_schema_options(),
        &ParquetMergeExecutionOptions {
            stats_fast_path: false,
            ..ordered_execution_options("event_id")
        },
    )
    .await?;
    assert!(enabled_report.fast_path_row_groups >= 2);
    assert_eq!(
        collect_int64_column(&read_parquet_batches(&enabled_output).await?, "event_id"),
        collect_int64_column(&read_parquet_batches(&disabled_output).await?, "event_id")
    );
    assert!(disabled_report.fast_path_row_groups == 0);

    let _ = tokio::fs::remove_file(first_path).await;
    let _ = tokio::fs::remove_file(second_path).await;
    let _ = tokio::fs::remove_file(enabled_output).await;
    let _ = tokio::fs::remove_file(disabled_output).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_mixed_null_row_groups_fall_back() -> Result<(), Box<dyn Error>> {
    let payload_fields: Fields = vec![Arc::new(Field::new("score", DataType::Int32, true))].into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int64, true),
        Field::new("org_id", DataType::Int64, false),
        Field::new("payload", DataType::Struct(payload_fields.clone()), true),
    ]));
    let payload = Arc::new(StructArray::new(
        payload_fields.clone(),
        vec![Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef],
        None,
    )) as ArrayRef;
    let mixed_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), None])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20])) as ArrayRef,
            payload,
        ],
    )?;
    let payload = Arc::new(StructArray::new(
        payload_fields,
        vec![Arc::new(Int32Array::from(vec![Some(3)])) as ArrayRef],
        None,
    )) as ArrayRef;
    let other_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(2)])) as ArrayRef,
            Arc::new(Int64Array::from(vec![30])) as ArrayRef,
            payload,
        ],
    )?;

    let writer_properties = WriterProperties::builder()
        .set_max_row_group_size(2)
        .build();
    let left_path = unique_path("ordered_mixed_nulls_left", "parquet");
    let right_path = unique_path("ordered_mixed_nulls_right", "parquet");
    let output_path = unique_path("ordered_mixed_nulls_output", "parquet");
    write_parquet_with_properties(
        &left_path,
        schema.clone(),
        mixed_batch,
        Some(writer_properties.clone()),
    )
    .await?;
    write_parquet_with_properties(&right_path, schema, other_batch, Some(writer_properties))
        .await?;

    let report = merge_payload_parquet_files_with_execution(
        &[left_path.clone(), right_path.clone()],
        &output_path,
        &PayloadMergeOptions::default(),
        &ordered_execution_options("event_id"),
    )
    .await?;
    assert!(report.fallback_batches > 0);
    assert_eq!(
        collect_int64_column(&read_parquet_batches(&output_path).await?, "event_id"),
        vec![Some(1), Some(2), None]
    );

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(right_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_handles_identical_payload_schemas() -> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, _, _) = sample_payload_inputs();
    let (first_schema, first_batch) = promote_payload_envelope_to_int64(
        left_schema.clone(),
        left_batch.clone(),
        &[1, 3],
        &[10, 20],
    );
    let (second_schema, second_batch) =
        promote_payload_envelope_to_int64(left_schema, left_batch, &[2, 4], &[30, 40]);
    let first_path = unique_path("ordered_identical_left", "parquet");
    let second_path = unique_path("ordered_identical_right", "parquet");
    let output_path = unique_path("ordered_identical_output", "parquet");

    write_parquet(&first_path, first_schema, first_batch).await?;
    write_parquet(&second_path, second_schema, second_batch).await?;

    let report = merge_payload_parquet_files_with_execution(
        &[first_path.clone(), second_path.clone()],
        &output_path,
        &payload_schema_options(),
        &ordered_execution_options("event_id"),
    )
    .await?;
    assert_eq!(report.rows, 4);
    assert!(report.ordered_merge_duration > Duration::default());
    assert!(report.input_batches > 0);
    assert!(report.output_batches > 0);

    let batches = read_parquet_batches(&output_path).await?;
    let event_ids = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(batch.schema().index_of("event_id").unwrap())
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("event_id is Int64")
                .iter()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(event_ids, vec![Some(1), Some(2), Some(3), Some(4)]);

    let _ = tokio::fs::remove_file(first_path).await;
    let _ = tokio::fs::remove_file(second_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_widens_payload_and_preserves_stable_ties() -> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, right_schema, right_batch) = sample_payload_inputs();
    let (left_schema, left_batch) =
        promote_payload_envelope_to_int64(left_schema, left_batch, &[1, 2], &[10, 20]);
    let (right_schema, right_batch) =
        promote_payload_envelope_to_int64(right_schema, right_batch, &[1, 3], &[30, 40]);
    let left_path = unique_path("ordered_mixed_left", "parquet");
    let right_path = unique_path("ordered_mixed_right", "parquet");
    let output_path = unique_path("ordered_mixed_output", "parquet");

    write_parquet(&left_path, left_schema, left_batch).await?;
    write_parquet(&right_path, right_schema, right_batch).await?;

    let report = merge_payload_parquet_files_with_execution(
        &[left_path.clone(), right_path.clone()],
        &output_path,
        &payload_schema_options(),
        &ordered_execution_options("event_id"),
    )
    .await?;
    assert_eq!(report.rows, 4);

    let batches = read_parquet_batches(&output_path).await?;
    let merged = &batches[0];
    let event_ids = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(batch.schema().index_of("event_id").unwrap())
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("event_id is Int64")
                .iter()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let org_ids = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(batch.schema().index_of("org_id").unwrap())
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("org_id is Int64")
                .iter()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(event_ids, vec![Some(1), Some(1), Some(2), Some(3)]);
    assert_eq!(org_ids, vec![Some(10), Some(30), Some(20), Some(40)]);

    let payload = merged
        .column(merged.schema().index_of("payload")?)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("payload is struct");
    let payload_fields = match payload.data_type() {
        DataType::Struct(fields) => fields,
        other => panic!("expected payload struct, got {other:?}"),
    };
    assert_eq!(payload_fields[0].data_type(), &DataType::Float64);

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(right_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_handles_interleaved_single_row_runs_with_schema_drift()
-> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, right_schema, right_batch) = sample_payload_inputs();
    let (first_left_schema, first_left_batch) = promote_payload_envelope_to_int64(
        left_schema.clone(),
        left_batch.clone(),
        &[1, 5],
        &[10, 50],
    );
    let (first_right_schema, first_right_batch) = promote_payload_envelope_to_int64(
        right_schema.clone(),
        right_batch.clone(),
        &[1, 6],
        &[30, 60],
    );
    let (second_left_schema, second_left_batch) =
        promote_payload_envelope_to_int64(left_schema, left_batch, &[2, 7], &[20, 70]);
    let (second_right_schema, second_right_batch) =
        promote_payload_envelope_to_int64(right_schema, right_batch, &[3, 8], &[40, 80]);

    let first_left_path = unique_path("ordered_interleaved_left_a", "parquet");
    let first_right_path = unique_path("ordered_interleaved_right_a", "parquet");
    let second_left_path = unique_path("ordered_interleaved_left_b", "parquet");
    let second_right_path = unique_path("ordered_interleaved_right_b", "parquet");
    let output_path = unique_path("ordered_interleaved_output", "parquet");

    write_parquet(&first_left_path, first_left_schema, first_left_batch).await?;
    write_parquet(&first_right_path, first_right_schema, first_right_batch).await?;
    write_parquet(&second_left_path, second_left_schema, second_left_batch).await?;
    write_parquet(&second_right_path, second_right_schema, second_right_batch).await?;

    let report = merge_payload_parquet_files_with_execution(
        &[
            first_left_path.clone(),
            first_right_path.clone(),
            second_left_path.clone(),
            second_right_path.clone(),
        ],
        &output_path,
        &payload_schema_options(),
        &ParquetMergeExecutionOptions {
            output_batch_rows: 3,
            output_row_group_rows: 3,
            ..ordered_execution_options("event_id")
        },
    )
    .await?;
    assert_eq!(report.rows, 8);
    assert_eq!(report.direct_batch_writes, 0);
    assert!(report.accumulator_flushes > 0);

    let batches = read_parquet_batches(&output_path).await?;
    assert_eq!(
        collect_int64_column(&batches, "event_id"),
        vec![
            Some(1),
            Some(1),
            Some(2),
            Some(3),
            Some(5),
            Some(6),
            Some(7),
            Some(8),
        ]
    );
    assert_eq!(
        collect_int64_column(&batches, "org_id"),
        vec![
            Some(10),
            Some(30),
            Some(20),
            Some(40),
            Some(50),
            Some(60),
            Some(70),
            Some(80),
        ]
    );

    let first_batch = batches.first().expect("ordered merge output has batches");
    let payload = first_batch
        .column(first_batch.schema().index_of("payload")?)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("payload is struct");
    let DataType::Struct(payload_fields) = payload.data_type() else {
        panic!("payload should remain a struct");
    };
    assert_eq!(payload_fields[0].data_type(), &DataType::Float64);
    assert!(payload_fields.iter().any(|field| field.name() == "amount"));

    let _ = tokio::fs::remove_file(first_left_path).await;
    let _ = tokio::fs::remove_file(first_right_path).await;
    let _ = tokio::fs::remove_file(second_left_path).await;
    let _ = tokio::fs::remove_file(second_right_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_dense_interleaved_rows_use_interleave_flush() -> Result<(), Box<dyn Error>> {
    let payload_fields: Fields = vec![Arc::new(Field::new("score", DataType::Int32, true))].into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int64, false),
        Field::new("org_id", DataType::Int64, false),
        Field::new("payload", DataType::Struct(payload_fields.clone()), true),
    ]));
    let mut paths = Vec::new();
    for source_index in 0..10 {
        let payload = Arc::new(StructArray::new(
            payload_fields.clone(),
            vec![Arc::new(Int32Array::from(vec![
                Some(source_index as i32),
                Some((source_index + 10) as i32),
            ])) as ArrayRef],
            None,
        )) as ArrayRef;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![
                    source_index as i64,
                    (source_index + 10) as i64,
                ])) as ArrayRef,
                Arc::new(Int64Array::from(vec![
                    (source_index * 10) as i64,
                    (source_index * 10 + 1) as i64,
                ])) as ArrayRef,
                payload,
            ],
        )?;
        let path = unique_path(
            &format!("ordered_dense_interleave_{source_index}"),
            "parquet",
        );
        write_parquet(&path, schema.clone(), batch).await?;
        paths.push(path);
    }
    let output_path = unique_path("ordered_dense_interleave_output", "parquet");

    let report = merge_payload_parquet_files_with_execution(
        &paths,
        &output_path,
        &payload_schema_options(),
        &ParquetMergeExecutionOptions {
            output_batch_rows: 10,
            output_row_group_rows: 10,
            parallelism: 2,
            ..ordered_execution_options("event_id")
        },
    )
    .await?;

    assert!(report.accumulator_interleave_flushes > 0);
    assert_eq!(
        collect_int64_column(&read_parquet_batches(&output_path).await?, "event_id"),
        (0..20).map(|value| Some(value as i64)).collect::<Vec<_>>()
    );

    for path in paths {
        let _ = tokio::fs::remove_file(path).await;
    }
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_rejects_unsorted_inputs() -> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, _, _) = sample_payload_inputs();
    let (left_schema, left_batch) =
        promote_payload_envelope_to_int64(left_schema, left_batch, &[2, 1], &[10, 20]);
    let left_path = unique_path("ordered_unsorted_left", "parquet");
    let output_path = unique_path("ordered_unsorted_output", "parquet");

    write_parquet(&left_path, left_schema, left_batch).await?;

    let error = merge_payload_parquet_files_with_execution(
        &[left_path.clone()],
        &output_path,
        &payload_schema_options(),
        &ordered_execution_options("event_id"),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(error.contains("not sorted ascending"));

    let _ = tokio::fs::remove_file(left_path).await;
    Ok(())
}

#[tokio::test]
async fn ordered_merge_rejects_missing_and_unsupported_ordering_fields()
-> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, _, _) = sample_payload_inputs();
    let (left_schema, left_batch) =
        promote_payload_envelope_to_int64(left_schema, left_batch, &[1, 2], &[10, 20]);
    let left_path = unique_path("ordered_missing_left", "parquet");
    let output_path = unique_path("ordered_missing_output", "parquet");
    write_parquet(&left_path, left_schema, left_batch).await?;

    let missing_error = merge_payload_parquet_files_with_execution(
        &[left_path.clone()],
        &output_path,
        &payload_schema_options(),
        &ordered_execution_options("missing"),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(missing_error.contains("missing from the merged schema"));

    let payload_fields: Fields = vec![Arc::new(Field::new("score", DataType::Int32, true))].into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int64, false),
        Field::new("is_active", DataType::Boolean, false),
        Field::new("payload", DataType::Struct(payload_fields.clone()), true),
    ]));
    let payload = Arc::new(StructArray::new(
        payload_fields,
        vec![Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef],
        None,
    )) as ArrayRef;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![true, false])) as ArrayRef,
            payload,
        ],
    )?;
    let unsupported_path = unique_path("ordered_unsupported_left", "parquet");
    write_parquet(&unsupported_path, schema, batch).await?;

    let unsupported_error = merge_payload_parquet_files_with_execution(
        &[unsupported_path.clone()],
        &output_path,
        &PayloadMergeOptions {
            payload_column: "payload".to_string(),
            widening_options: WideningOptions::default(),
        },
        &ParquetMergeExecutionOptions {
            ordering_field: Some("is_active".to_string()),
            ..ordered_execution_options("event_id")
        },
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(unsupported_error.contains("unsupported type Boolean"));

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(unsupported_path).await;
    Ok(())
}

#[tokio::test]
async fn merge_payload_parquet_files_merges_many_inputs_with_compiled_plan()
-> Result<(), Box<dyn Error>> {
    let (left_schema, left_batch, right_schema, right_batch) = sample_payload_inputs();
    let left_path = unique_path("merge_left", "parquet");
    let right_path = unique_path("merge_right", "parquet");
    let output_path = unique_path("merge_output", "parquet");

    write_parquet(&left_path, left_schema, left_batch).await?;
    write_parquet(&right_path, right_schema, right_batch).await?;

    let report = merge_payload_parquet_files(
        &[left_path.clone(), right_path.clone()],
        &output_path,
        &payload_schema_options(),
    )
    .await?;
    assert_eq!(report.rows, 4);
    assert_eq!(report.ordered_merge_duration, Duration::default());
    assert!(report.input_batches > 0);
    assert!(report.output_batches > 0);

    let merged_batches = read_parquet_batches(&output_path).await?;
    let payload = merged_batches[0]
        .column(merged_batches[0].schema().index_of("payload")?)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("payload is a struct");
    let payload_fields = match payload.data_type() {
        DataType::Struct(fields) => fields,
        other => panic!("expected payload struct, got {other:?}"),
    };
    assert_eq!(payload_fields[0].data_type(), &DataType::Float64);

    let _ = tokio::fs::remove_file(left_path).await;
    let _ = tokio::fs::remove_file(right_path).await;
    let _ = tokio::fs::remove_file(output_path).await;
    Ok(())
}
