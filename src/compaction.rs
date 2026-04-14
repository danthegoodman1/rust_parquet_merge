use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fs::File as StdFile;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::{
    PayloadMergeOptions, WideningOptions, is_primitive_or_string, merge_payload_schemas_pair,
    to_parquet_error, widen_data_type,
};
use arrow::array::new_null_array;
use arrow::compute::cast;
use arrow_array::builder::{
    ArrayBuilder, BooleanBuilder, Float32Builder, Float64Builder, Int8Builder, Int16Builder,
    Int32Builder, Int64Builder, LargeListBuilder, LargeStringBuilder, ListBuilder, NullBuilder,
    StringBuilder, UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
};
use arrow_array::{Array, ArrayRef, LargeListArray, ListArray, RecordBatch, StructArray};
use arrow_buffer::NullBufferBuilder;
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use futures_util::StreamExt;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_writer::AsyncArrowWriter;
use simd_json::borrowed::Value as BorrowedValue;
use simd_json::prelude::{TypedScalarValue, ValueAsArray, ValueAsObject, ValueAsScalar};
use simd_json::{Buffers, to_borrowed_value_with_buffers};
use tokio::fs::File;
use tokio::io::BufWriter;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionOptions {
    pub envelope_fields: Vec<String>,
    pub payload_column: String,
    pub widening_options: WideningOptions,
    pub batch_rows: usize,
    pub scan_parallelism: Option<usize>,
    pub read_buffer_bytes: usize,
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            envelope_fields: Vec::new(),
            payload_column: "payload".to_string(),
            widening_options: WideningOptions::default(),
            batch_rows: 8_192,
            scan_parallelism: std::thread::available_parallelism()
                .ok()
                .map(std::num::NonZeroUsize::get)
                .or(Some(1)),
            read_buffer_bytes: 1 << 20,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompactionReport {
    pub rows: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub planning_duration: Duration,
    pub execution_duration: Duration,
    pub total_duration: Duration,
    pub peak_rss_bytes: u64,
    pub planning_threads_used: usize,
    pub planning_unique_shapes: u64,
    pub planning_shape_cache_hits: u64,
    pub planning_shape_cache_misses: u64,
}

#[derive(Debug, Default)]
pub struct ExecutionScratch {
    buffers: Vec<Vec<ArrayRef>>,
}

impl ExecutionScratch {
    fn take_buffer(&mut self, depth: usize, capacity: usize) -> Vec<ArrayRef> {
        if self.buffers.len() <= depth {
            self.buffers.resize_with(depth + 1, Vec::new);
        }

        let mut buffer = std::mem::take(&mut self.buffers[depth]);
        if buffer.capacity() < capacity {
            buffer.reserve(capacity - buffer.capacity());
        }
        buffer.clear();
        buffer
    }

    fn return_buffer(&mut self, depth: usize, mut buffer: Vec<ArrayRef>) {
        if self.buffers.len() <= depth {
            self.buffers.resize_with(depth + 1, Vec::new);
        }
        buffer.clear();
        self.buffers[depth] = buffer;
    }
}

#[derive(Debug)]
pub struct CompiledPayloadPlan {
    pub output_schema: SchemaRef,
    pub source_adapter_cache: HashMap<String, Arc<CompiledSourceAdapter>>,
    pub scratch: ExecutionScratch,
}

impl CompiledPayloadPlan {
    pub fn new(output_schema: SchemaRef) -> Self {
        Self {
            output_schema,
            source_adapter_cache: HashMap::new(),
            scratch: ExecutionScratch::default(),
        }
    }

    pub fn from_payload_schemas<'a, I>(
        schemas: I,
        options: &PayloadMergeOptions,
    ) -> Result<Self, String>
    where
        I: IntoIterator<Item = &'a Schema>,
    {
        build_compiled_payload_plan(schemas, options)
    }

    pub fn source_adapter_for_schema(
        &mut self,
        source_schema: &Schema,
    ) -> Result<Arc<CompiledSourceAdapter>, String> {
        let fingerprint = schema_fingerprint(source_schema);
        if let Some(adapter) = self.source_adapter_cache.get(&fingerprint) {
            return Ok(adapter.clone());
        }

        let adapter = Arc::new(CompiledSourceAdapter::compile(
            source_schema,
            self.output_schema.as_ref(),
        )?);
        self.source_adapter_cache
            .insert(fingerprint, adapter.clone());
        Ok(adapter)
    }

    pub fn adapt_batch(
        &mut self,
        batch: &RecordBatch,
    ) -> Result<RecordBatch, parquet::errors::ParquetError> {
        let adapter = self
            .source_adapter_for_schema(batch.schema().as_ref())
            .map_err(to_parquet_error)?;
        adapter
            .adapt_batch(batch, &self.output_schema, &mut self.scratch)
            .map_err(to_parquet_error)
    }
}

#[derive(Debug)]
pub struct CompiledSourceAdapter {
    pub fingerprint: String,
    pub identical_schema: bool,
    operations: Vec<CompiledColumnOperation>,
}

impl CompiledSourceAdapter {
    fn compile(source_schema: &Schema, target_schema: &Schema) -> Result<Self, String> {
        let fingerprint = schema_fingerprint(source_schema);
        if source_schema == target_schema {
            return Ok(Self {
                fingerprint,
                identical_schema: true,
                operations: Vec::new(),
            });
        }

        let source_lookup: HashMap<&str, usize> = source_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| (field.name().as_str(), index))
            .collect();

        let mut operations = Vec::with_capacity(target_schema.fields().len());
        for target_field in target_schema.fields() {
            let source_index = source_lookup.get(target_field.name().as_str()).copied();
            let adapter = match source_index {
                Some(index) => compile_type_adapter(
                    source_schema.field(index).data_type(),
                    target_field.data_type(),
                )?,
                None => CompiledTypeAdapter::NullFill {
                    target_type: target_field.data_type().clone(),
                },
            };
            operations.push(CompiledColumnOperation {
                source_index,
                target_field: target_field.clone(),
                adapter,
            });
        }

        Ok(Self {
            fingerprint,
            identical_schema: false,
            operations,
        })
    }

    fn adapt_batch(
        &self,
        batch: &RecordBatch,
        target_schema: &SchemaRef,
        scratch: &mut ExecutionScratch,
    ) -> Result<RecordBatch, String> {
        if self.identical_schema {
            if batch.schema().as_ref() == target_schema.as_ref() {
                return Ok(batch.clone());
            }
            return RecordBatch::try_new(target_schema.clone(), batch.columns().to_vec())
                .map_err(|error| format!("Arrow error: {error}"));
        }

        let mut columns = scratch.take_buffer(0, self.operations.len());
        for operation in &self.operations {
            columns.push(operation.apply(batch, scratch, 1)?);
        }

        let batch = RecordBatch::try_new(target_schema.clone(), columns.clone())
            .map_err(|error| format!("Arrow error: {error}"))?;
        scratch.return_buffer(0, columns);
        Ok(batch)
    }
}

#[derive(Clone, Debug)]
struct CompiledColumnOperation {
    source_index: Option<usize>,
    target_field: FieldRef,
    adapter: CompiledTypeAdapter,
}

impl CompiledColumnOperation {
    fn apply(
        &self,
        batch: &RecordBatch,
        scratch: &mut ExecutionScratch,
        depth: usize,
    ) -> Result<ArrayRef, String> {
        match self.source_index {
            Some(index) => self
                .adapter
                .apply_existing(batch.column(index), scratch, depth),
            None => Ok(new_null_array(
                self.target_field.data_type(),
                batch.num_rows(),
            )),
        }
    }
}

#[derive(Clone, Debug)]
enum CompiledTypeAdapter {
    PassThrough,
    PrimitiveCast { target_type: DataType },
    NullFill { target_type: DataType },
    StructAdapter(CompiledStructAdapter),
    ListAdapter(CompiledListAdapter),
    LargeListAdapter(CompiledListAdapter),
}

impl CompiledTypeAdapter {
    fn apply_existing(
        &self,
        array: &ArrayRef,
        scratch: &mut ExecutionScratch,
        depth: usize,
    ) -> Result<ArrayRef, String> {
        match self {
            Self::PassThrough => Ok(array.clone()),
            Self::PrimitiveCast { target_type } => {
                cast(array.as_ref(), target_type).map_err(|error| {
                    format!(
                        "failed primitive cast from {:?} to {:?}: {error}",
                        array.data_type(),
                        target_type
                    )
                })
            }
            Self::NullFill { target_type } => Ok(new_null_array(target_type, array.len())),
            Self::StructAdapter(adapter) => adapter.apply(array, scratch, depth),
            Self::ListAdapter(adapter) => adapter.apply_list(array, scratch, depth),
            Self::LargeListAdapter(adapter) => adapter.apply_large_list(array, scratch, depth),
        }
    }
}

#[derive(Clone, Debug)]
struct CompiledStructAdapter {
    target_fields: Fields,
    children: Vec<CompiledStructChild>,
}

#[derive(Clone, Debug)]
struct CompiledStructChild {
    source_index: Option<usize>,
    target_field: FieldRef,
    adapter: CompiledTypeAdapter,
}

impl CompiledStructAdapter {
    fn apply(
        &self,
        array: &ArrayRef,
        scratch: &mut ExecutionScratch,
        depth: usize,
    ) -> Result<ArrayRef, String> {
        let source_struct = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| format!("expected StructArray, got {:?}", array.data_type()))?;

        let mut children = scratch.take_buffer(depth, self.children.len());
        for child in &self.children {
            match child.source_index {
                Some(index) => children.push(child.adapter.apply_existing(
                    source_struct.column(index),
                    scratch,
                    depth + 1,
                )?),
                None => children.push(new_null_array(
                    child.target_field.data_type(),
                    source_struct.len(),
                )),
            }
        }

        let coerced = Arc::new(StructArray::new(
            self.target_fields.clone(),
            children.clone(),
            source_struct.nulls().cloned(),
        )) as ArrayRef;
        scratch.return_buffer(depth, children);
        Ok(coerced)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ListKind {
    List,
    LargeList,
}

#[derive(Clone, Debug)]
struct CompiledListAdapter {
    target_type: DataType,
    target_field: FieldRef,
    value_adapter: Box<CompiledTypeAdapter>,
    source_kind: ListKind,
}

impl CompiledListAdapter {
    fn apply_list(
        &self,
        array: &ArrayRef,
        scratch: &mut ExecutionScratch,
        depth: usize,
    ) -> Result<ArrayRef, String> {
        match self.source_kind {
            ListKind::List => {
                let source_list = array
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| format!("expected ListArray, got {:?}", array.data_type()))?;

                let values =
                    self.value_adapter
                        .apply_existing(&source_list.values(), scratch, depth + 1)?;
                Ok(Arc::new(ListArray::new(
                    self.target_field.clone(),
                    source_list.offsets().clone(),
                    values,
                    source_list.nulls().cloned(),
                )) as ArrayRef)
            }
            ListKind::LargeList => {
                let source_list =
                    array
                        .as_any()
                        .downcast_ref::<LargeListArray>()
                        .ok_or_else(|| {
                            format!("expected LargeListArray, got {:?}", array.data_type())
                        })?;
                let values =
                    self.value_adapter
                        .apply_existing(&source_list.values(), scratch, depth + 1)?;
                let intermediate = Arc::new(LargeListArray::new(
                    self.target_field.clone(),
                    source_list.offsets().clone(),
                    values,
                    source_list.nulls().cloned(),
                )) as ArrayRef;
                cast(intermediate.as_ref(), &self.target_type).map_err(|error| {
                    format!(
                        "failed large-list-to-list cast from {:?} to {:?}: {error}",
                        array.data_type(),
                        self.target_type
                    )
                })
            }
        }
    }

    fn apply_large_list(
        &self,
        array: &ArrayRef,
        scratch: &mut ExecutionScratch,
        depth: usize,
    ) -> Result<ArrayRef, String> {
        match self.source_kind {
            ListKind::List => {
                let source_list = array
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| format!("expected ListArray, got {:?}", array.data_type()))?;
                let values =
                    self.value_adapter
                        .apply_existing(&source_list.values(), scratch, depth + 1)?;
                let intermediate = Arc::new(ListArray::new(
                    self.target_field.clone(),
                    source_list.offsets().clone(),
                    values,
                    source_list.nulls().cloned(),
                )) as ArrayRef;
                cast(intermediate.as_ref(), &self.target_type).map_err(|error| {
                    format!(
                        "failed list-to-large-list cast from {:?} to {:?}: {error}",
                        array.data_type(),
                        self.target_type
                    )
                })
            }
            ListKind::LargeList => {
                let source_list =
                    array
                        .as_any()
                        .downcast_ref::<LargeListArray>()
                        .ok_or_else(|| {
                            format!("expected LargeListArray, got {:?}", array.data_type())
                        })?;
                let values =
                    self.value_adapter
                        .apply_existing(&source_list.values(), scratch, depth + 1)?;
                Ok(Arc::new(LargeListArray::new(
                    self.target_field.clone(),
                    source_list.offsets().clone(),
                    values,
                    source_list.nulls().cloned(),
                )) as ArrayRef)
            }
        }
    }
}

pub fn build_compiled_payload_plan<'a, I>(
    schemas: I,
    options: &PayloadMergeOptions,
) -> Result<CompiledPayloadPlan, String>
where
    I: IntoIterator<Item = &'a Schema>,
{
    let output_schema = Arc::new(merge_payload_schemas_many(schemas, options)?);
    Ok(CompiledPayloadPlan::new(output_schema))
}

pub async fn merge_payload_parquet_files(
    input_paths: &[PathBuf],
    output_path: &Path,
    options: &PayloadMergeOptions,
) -> Result<CompactionReport, Box<dyn Error>> {
    if input_paths.is_empty() {
        return Err(io_error("at least one parquet input path is required").into());
    }

    let total_start = Instant::now();
    let planning_start = Instant::now();

    let mut input_bytes = 0_u64;
    let mut builders = Vec::with_capacity(input_paths.len());
    let mut schemas = Vec::with_capacity(input_paths.len());
    for input_path in input_paths {
        input_bytes += std::fs::metadata(input_path)?.len();
        let file = File::open(input_path).await?;
        let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
        schemas.push(builder.schema().clone());
        builders.push(builder);
    }

    let mut plan = CompiledPayloadPlan::from_payload_schemas(
        schemas.iter().map(|schema| schema.as_ref()),
        options,
    )?;
    for schema in &schemas {
        let _ = plan.source_adapter_for_schema(schema.as_ref())?;
    }

    let planning_duration = planning_start.elapsed();
    let execution_start = Instant::now();

    let output_file = File::create(output_path).await?;
    let output_file = BufWriter::with_capacity(1 << 20, output_file);
    let mut writer = AsyncArrowWriter::try_new(output_file, plan.output_schema.clone(), None)?;

    let mut rows = 0_u64;
    for builder in builders {
        let mut stream = builder.build()?;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            rows += batch.num_rows() as u64;
            let adjusted_batch = plan.adapt_batch(&batch)?;
            writer.write(&adjusted_batch).await?;
        }
    }

    writer.close().await?;

    let execution_duration = execution_start.elapsed();
    let output_bytes = std::fs::metadata(output_path)?.len();

    Ok(CompactionReport {
        rows,
        input_bytes,
        output_bytes,
        planning_duration,
        execution_duration,
        total_duration: total_start.elapsed(),
        peak_rss_bytes: peak_rss_bytes(),
        planning_threads_used: 0,
        planning_unique_shapes: 0,
        planning_shape_cache_hits: 0,
        planning_shape_cache_misses: 0,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct NdjsonPlanningStats {
    threads_used: usize,
    unique_shapes: u64,
    shape_cache_hits: u64,
    shape_cache_misses: u64,
}

#[derive(Clone, Debug)]
struct NdjsonPlanningResult {
    schema: SchemaRef,
    stats: NdjsonPlanningStats,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ShapeKey(String);

#[derive(Clone, Debug, PartialEq, Eq)]
struct DiscoveredField {
    name: String,
    data_type: DiscoveredType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DiscoveredType {
    nullable: bool,
    kind: DiscoveredKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DiscoveredKind {
    Null,
    Boolean,
    Int64,
    UInt64,
    Float64,
    Utf8,
    Struct(Vec<DiscoveredField>),
    List(Box<DiscoveredType>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CachedDiscovery {
    envelope_types: Vec<Option<DiscoveredType>>,
    payload_type: DiscoveredType,
}

#[derive(Clone, Debug, Default)]
struct SchemaAccumulator {
    envelope_types: Vec<Option<DiscoveredType>>,
    envelope_missing: Vec<bool>,
    payload_type: Option<DiscoveredType>,
}

#[derive(Clone, Debug, Default)]
struct FileDiscoveryResult {
    accumulator: SchemaAccumulator,
    rows: u64,
    unique_shapes: u64,
    shape_cache_hits: u64,
    shape_cache_misses: u64,
}

#[derive(Default)]
struct NdjsonScanState {
    line_buffer: Vec<u8>,
    parser_buffers: Buffers,
    shape_buffer: String,
}

impl NdjsonScanState {
    fn with_capacity(read_buffer_bytes: usize) -> Self {
        Self {
            line_buffer: Vec::with_capacity(read_buffer_bytes),
            parser_buffers: Buffers::default(),
            shape_buffer: String::with_capacity(256),
        }
    }
}

#[derive(Debug, Default)]
struct NdjsonAppendScratch {
    seen_buffers: Vec<Vec<bool>>,
}

impl NdjsonAppendScratch {
    fn take_seen(&mut self, depth: usize, capacity: usize) -> Vec<bool> {
        if self.seen_buffers.len() <= depth {
            self.seen_buffers.resize_with(depth + 1, Vec::new);
        }

        let mut buffer = std::mem::take(&mut self.seen_buffers[depth]);
        if buffer.len() < capacity {
            buffer.resize(capacity, false);
        } else {
            buffer.truncate(capacity);
            buffer.fill(false);
        }
        buffer
    }

    fn return_seen(&mut self, depth: usize, mut buffer: Vec<bool>) {
        if self.seen_buffers.len() <= depth {
            self.seen_buffers.resize_with(depth + 1, Vec::new);
        }
        buffer.clear();
        self.seen_buffers[depth] = buffer;
    }
}

#[derive(Clone, Debug)]
struct CompiledNdjsonPlan {
    envelope_ops: Vec<CompiledNdjsonFieldOp>,
    payload_op: CompiledNdjsonStructOp,
    payload_builder_index: usize,
    top_level_lookup: HashMap<String, RootDispatch>,
}

#[derive(Clone, Copy, Debug)]
enum RootDispatch {
    Envelope(usize),
    Payload(usize),
}

#[derive(Clone, Debug)]
struct CompiledNdjsonFieldOp {
    pub name: String,
    builder_index: usize,
    value_op: CompiledNdjsonValueOp,
}

#[derive(Clone, Debug)]
enum CompiledNdjsonValueOp {
    Null,
    Boolean,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Utf8,
    LargeUtf8,
    Struct(CompiledNdjsonStructOp),
    List(Box<CompiledNdjsonValueOp>),
    LargeList(Box<CompiledNdjsonValueOp>),
}

#[derive(Clone, Debug)]
struct CompiledNdjsonStructOp {
    fields: Vec<CompiledNdjsonFieldOp>,
    lookup: HashMap<String, usize>,
}

#[derive(Debug)]
struct NdjsonBatchBuilder {
    batch_rows: usize,
    plan: CompiledNdjsonPlan,
    root: NdjsonStructBuilder,
    scratch: NdjsonAppendScratch,
}

#[derive(Debug)]
struct NdjsonStructBuilder {
    fields: Fields,
    children: Vec<NdjsonValueBuilder>,
    nulls: NullBufferBuilder,
}

#[derive(Debug)]
enum NdjsonValueBuilder {
    Null(NullBuilder),
    Boolean(BooleanBuilder),
    Int8(Int8Builder),
    Int16(Int16Builder),
    Int32(Int32Builder),
    Int64(Int64Builder),
    UInt8(UInt8Builder),
    UInt16(UInt16Builder),
    UInt32(UInt32Builder),
    UInt64(UInt64Builder),
    Float32(Float32Builder),
    Float64(Float64Builder),
    Utf8(StringBuilder),
    LargeUtf8(LargeStringBuilder),
    Struct(NdjsonStructBuilder),
    List(Box<ListBuilder<NdjsonValueBuilder>>),
    LargeList(Box<LargeListBuilder<NdjsonValueBuilder>>),
}

pub fn discover_ndjson_schema_from_paths(
    input_paths: &[PathBuf],
    options: &CompactionOptions,
) -> Result<SchemaRef, Box<dyn Error>> {
    Ok(discover_ndjson_schema_details_from_paths(input_paths, options)?.schema)
}

pub async fn compact_ndjson_to_parquet(
    input_paths: &[PathBuf],
    output_path: &Path,
    options: &CompactionOptions,
) -> Result<CompactionReport, Box<dyn Error>> {
    validate_compaction_options(options).map_err(io_error)?;
    if input_paths.is_empty() {
        return Err(io_error("at least one NDJSON input path is required").into());
    }

    let total_start = Instant::now();
    let planning_start = Instant::now();
    let planning = discover_ndjson_schema_details_from_paths(input_paths, options)?;
    let planning_duration = planning_start.elapsed();

    let execution_start = Instant::now();
    let input_bytes = input_paths
        .iter()
        .map(std::fs::metadata)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|metadata| metadata.len())
        .sum();

    let output_file = File::create(output_path).await?;
    let output_file = BufWriter::with_capacity(1 << 20, output_file);
    let mut writer = AsyncArrowWriter::try_new(output_file, planning.schema.clone(), None)?;
    let plan = CompiledNdjsonPlan::compile(planning.schema.as_ref(), options).map_err(io_error)?;
    let mut batch_builder =
        NdjsonBatchBuilder::new(planning.schema.clone(), plan, options.batch_rows)
            .map_err(io_error)?;
    let mut rows = 0_u64;

    for input_path in input_paths {
        let file = StdFile::open(input_path)?;
        let mut reader = BufReader::with_capacity(options.read_buffer_bytes, file);
        let mut scan_state = NdjsonScanState::with_capacity(options.read_buffer_bytes);

        while let Some(value) = read_next_ndjson_value(
            &mut reader,
            &mut scan_state.line_buffer,
            &mut scan_state.parser_buffers,
        )
        .map_err(io_error)?
        {
            batch_builder.append_record(&value).map_err(io_error)?;
            rows += 1;

            if batch_builder.is_full() {
                writer.write(&batch_builder.finish_batch()).await?;
            }
        }
    }

    if !batch_builder.is_empty() {
        writer.write(&batch_builder.finish_batch()).await?;
    }

    writer.close().await?;

    let execution_duration = execution_start.elapsed();
    let output_bytes = std::fs::metadata(output_path)?.len();

    Ok(CompactionReport {
        rows,
        input_bytes,
        output_bytes,
        planning_duration,
        execution_duration,
        total_duration: total_start.elapsed(),
        peak_rss_bytes: peak_rss_bytes(),
        planning_threads_used: planning.stats.threads_used,
        planning_unique_shapes: planning.stats.unique_shapes,
        planning_shape_cache_hits: planning.stats.shape_cache_hits,
        planning_shape_cache_misses: planning.stats.shape_cache_misses,
    })
}

fn discover_ndjson_schema_details_from_paths(
    input_paths: &[PathBuf],
    options: &CompactionOptions,
) -> Result<NdjsonPlanningResult, Box<dyn Error>> {
    discover_ndjson_schema_details_with_settings(input_paths, options, true)
}

fn discover_ndjson_schema_details_with_settings(
    input_paths: &[PathBuf],
    options: &CompactionOptions,
    cache_enabled: bool,
) -> Result<NdjsonPlanningResult, Box<dyn Error>> {
    validate_compaction_options(options).map_err(io_error)?;
    if input_paths.is_empty() {
        return Err(io_error("at least one NDJSON input path is required").into());
    }

    let thread_target = options
        .scan_parallelism
        .unwrap_or_else(default_scan_parallelism)
        .max(1)
        .min(input_paths.len().max(1));

    let file_results = if thread_target == 1 || input_paths.len() == 1 {
        let mut results = Vec::with_capacity(input_paths.len());
        for path in input_paths {
            results.push(scan_ndjson_file(path, options, cache_enabled).map_err(io_error)?);
        }
        results
    } else {
        discover_ndjson_files_in_parallel(input_paths, options, thread_target, cache_enabled)
            .map_err(io_error)?
    };

    let schema =
        Arc::new(reduce_discovery_results(file_results.as_slice(), options).map_err(io_error)?);
    let stats = NdjsonPlanningStats {
        threads_used: thread_target,
        unique_shapes: file_results.iter().map(|result| result.unique_shapes).sum(),
        shape_cache_hits: file_results
            .iter()
            .map(|result| result.shape_cache_hits)
            .sum(),
        shape_cache_misses: file_results
            .iter()
            .map(|result| result.shape_cache_misses)
            .sum(),
    };

    Ok(NdjsonPlanningResult { schema, stats })
}

fn discover_ndjson_files_in_parallel(
    input_paths: &[PathBuf],
    options: &CompactionOptions,
    thread_target: usize,
    cache_enabled: bool,
) -> Result<Vec<FileDiscoveryResult>, String> {
    let next_index = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, Result<FileDiscoveryResult, String>)>> = Mutex::new(Vec::new());

    thread::scope(|scope| {
        for _ in 0..thread_target {
            let next_index = &next_index;
            let results = &results;
            scope.spawn(move || {
                loop {
                    let index = next_index.fetch_add(1, Ordering::SeqCst);
                    if index >= input_paths.len() {
                        break;
                    }

                    let result = scan_ndjson_file(&input_paths[index], options, cache_enabled);
                    results
                        .lock()
                        .expect("results mutex is available")
                        .push((index, result));
                }
            });
        }
    });

    let mut ordered = results
        .into_inner()
        .map_err(|_| "failed to collect NDJSON discovery results".to_string())?;
    ordered.sort_unstable_by_key(|(index, _)| *index);

    let mut file_results = Vec::with_capacity(ordered.len());
    for (_, result) in ordered {
        file_results.push(result?);
    }
    Ok(file_results)
}

fn scan_ndjson_file(
    input_path: &Path,
    options: &CompactionOptions,
    cache_enabled: bool,
) -> Result<FileDiscoveryResult, String> {
    let file = StdFile::open(input_path)
        .map_err(|error| format!("failed to open `{}`: {error}", input_path.display()))?;
    let mut reader = BufReader::with_capacity(options.read_buffer_bytes, file);
    let envelope_lookup: HashMap<&str, usize> = options
        .envelope_fields
        .iter()
        .enumerate()
        .map(|(index, field)| (field.as_str(), index))
        .collect();
    let mut scan_state = NdjsonScanState::with_capacity(options.read_buffer_bytes);
    let mut shape_cache: HashMap<ShapeKey, Arc<CachedDiscovery>> = HashMap::new();
    let mut result = FileDiscoveryResult {
        accumulator: SchemaAccumulator {
            envelope_types: vec![None; options.envelope_fields.len()],
            envelope_missing: vec![false; options.envelope_fields.len()],
            payload_type: None,
        },
        ..FileDiscoveryResult::default()
    };

    while let Some(value) = read_next_ndjson_value(
        &mut reader,
        &mut scan_state.line_buffer,
        &mut scan_state.parser_buffers,
    )
    .map_err(|error| format!("failed to parse `{}`: {error}", input_path.display()))?
    {
        let object = value.as_object().ok_or_else(|| {
            format!(
                "NDJSON record in `{}` must be an object",
                input_path.display()
            )
        })?;

        let discovery = if cache_enabled {
            let shape_key = compute_shape_key(&value, &mut scan_state.shape_buffer);
            if let Some(cached) = shape_cache.get(&shape_key) {
                result.shape_cache_hits += 1;
                cached.clone()
            } else {
                result.shape_cache_misses += 1;
                let discovered = Arc::new(discover_record(object, &envelope_lookup, options)?);
                shape_cache.insert(shape_key, discovered.clone());
                discovered
            }
        } else {
            result.shape_cache_misses += 1;
            Arc::new(discover_record(object, &envelope_lookup, options)?)
        };

        merge_record_into_accumulator(&mut result.accumulator, &discovery, options)?;
        result.rows += 1;
    }

    result.unique_shapes = if cache_enabled {
        shape_cache.len() as u64
    } else {
        result.shape_cache_misses
    };
    Ok(result)
}

fn discover_record(
    object: &simd_json::borrowed::Object<'_>,
    envelope_lookup: &HashMap<&str, usize>,
    options: &CompactionOptions,
) -> Result<CachedDiscovery, String> {
    let mut envelope_types = vec![None; envelope_lookup.len()];
    let mut payload_fields = Vec::new();

    for (key, value) in object.iter() {
        if let Some(index) = envelope_lookup.get(key.as_ref()) {
            envelope_types[*index] =
                Some(discover_borrowed_type(value, &options.widening_options)?);
        } else {
            payload_fields.push(DiscoveredField {
                name: key.to_string(),
                data_type: discover_borrowed_type(value, &options.widening_options)?,
            });
        }
    }

    payload_fields.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    Ok(CachedDiscovery {
        envelope_types,
        payload_type: DiscoveredType {
            nullable: false,
            kind: DiscoveredKind::Struct(payload_fields),
        },
    })
}

fn discover_borrowed_type(
    value: &BorrowedValue<'_>,
    options: &WideningOptions,
) -> Result<DiscoveredType, String> {
    if value.is_null() {
        return Ok(DiscoveredType {
            nullable: true,
            kind: DiscoveredKind::Null,
        });
    }

    if value.as_bool().is_some() {
        return Ok(DiscoveredType {
            nullable: false,
            kind: DiscoveredKind::Boolean,
        });
    }

    if value.as_i64().is_some() {
        return Ok(DiscoveredType {
            nullable: false,
            kind: DiscoveredKind::Int64,
        });
    }

    if value.as_u64().is_some() {
        return Ok(DiscoveredType {
            nullable: false,
            kind: DiscoveredKind::UInt64,
        });
    }

    if value.as_f64().is_some() {
        return Ok(DiscoveredType {
            nullable: false,
            kind: DiscoveredKind::Float64,
        });
    }

    if value.as_str().is_some() {
        return Ok(DiscoveredType {
            nullable: false,
            kind: DiscoveredKind::Utf8,
        });
    }

    if let Some(values) = value.as_array() {
        let mut element_type: Option<DiscoveredType> = None;
        for element in values.iter() {
            let discovered = discover_borrowed_type(element, options)?;
            element_type = Some(match element_type {
                Some(current) => {
                    merge_discovered_types("payload[]", &current, &discovered, options)?
                }
                None => discovered,
            });
        }

        return Ok(DiscoveredType {
            nullable: false,
            kind: DiscoveredKind::List(Box::new(element_type.unwrap_or(DiscoveredType {
                nullable: true,
                kind: DiscoveredKind::Null,
            }))),
        });
    }

    if let Some(object) = value.as_object() {
        let mut fields = Vec::with_capacity(object.len());
        for (key, child) in object.iter() {
            fields.push(DiscoveredField {
                name: key.to_string(),
                data_type: discover_borrowed_type(child, options)?,
            });
        }
        fields.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        return Ok(DiscoveredType {
            nullable: false,
            kind: DiscoveredKind::Struct(fields),
        });
    }

    Err(format!(
        "unsupported NDJSON value kind during discovery: {}",
        borrowed_json_kind(value)
    ))
}

fn merge_record_into_accumulator(
    accumulator: &mut SchemaAccumulator,
    discovery: &CachedDiscovery,
    options: &CompactionOptions,
) -> Result<(), String> {
    for (index, field_name) in options.envelope_fields.iter().enumerate() {
        match &discovery.envelope_types[index] {
            Some(discovered) => {
                if let Some(existing) = accumulator.envelope_types[index].as_ref() {
                    let merged = merge_discovered_types(
                        field_name,
                        existing,
                        discovered,
                        &options.widening_options,
                    )?;
                    let merged_nullable = merged.nullable;
                    accumulator.envelope_types[index] = Some(with_nullable(
                        merged,
                        accumulator.envelope_missing[index] || merged_nullable,
                    ));
                } else {
                    accumulator.envelope_types[index] = Some(with_nullable(
                        discovered.clone(),
                        accumulator.envelope_missing[index] || discovered.nullable,
                    ));
                }
            }
            None => {
                accumulator.envelope_missing[index] = true;
                if let Some(existing) = accumulator.envelope_types[index].as_mut() {
                    existing.nullable = true;
                }
            }
        }
    }

    accumulator.payload_type = Some(match accumulator.payload_type.as_ref() {
        Some(existing) => merge_discovered_types(
            "payload",
            existing,
            &discovery.payload_type,
            &options.widening_options,
        )?,
        None => discovery.payload_type.clone(),
    });

    Ok(())
}

fn reduce_discovery_results(
    results: &[FileDiscoveryResult],
    options: &CompactionOptions,
) -> Result<Schema, String> {
    let mut accumulator = SchemaAccumulator {
        envelope_types: vec![None; options.envelope_fields.len()],
        envelope_missing: vec![false; options.envelope_fields.len()],
        payload_type: None,
    };

    for result in results {
        merge_accumulators(&mut accumulator, &result.accumulator, options)?;
    }

    let mut fields = Vec::with_capacity(options.envelope_fields.len() + 1);
    for (index, field_name) in options.envelope_fields.iter().enumerate() {
        let discovered = accumulator.envelope_types[index]
            .clone()
            .unwrap_or(DiscoveredType {
                nullable: true,
                kind: DiscoveredKind::Null,
            });
        fields.push(Arc::new(Field::new(
            field_name,
            discovered_to_arrow_data_type(&discovered),
            accumulator.envelope_missing[index] || discovered.nullable,
        )));
    }

    let payload = accumulator.payload_type.unwrap_or(DiscoveredType {
        nullable: false,
        kind: DiscoveredKind::Struct(Vec::new()),
    });
    fields.push(Arc::new(Field::new(
        &options.payload_column,
        discovered_to_arrow_data_type(&payload),
        true,
    )));

    Ok(Schema::new(fields))
}

fn merge_accumulators(
    target: &mut SchemaAccumulator,
    source: &SchemaAccumulator,
    options: &CompactionOptions,
) -> Result<(), String> {
    for (index, field_name) in options.envelope_fields.iter().enumerate() {
        match &source.envelope_types[index] {
            Some(source_type) => {
                if let Some(existing) = target.envelope_types[index].as_ref() {
                    let merged = merge_discovered_types(
                        field_name,
                        existing,
                        source_type,
                        &options.widening_options,
                    )?;
                    let merged_nullable = merged.nullable;
                    target.envelope_types[index] = Some(with_nullable(
                        merged,
                        target.envelope_missing[index]
                            || source.envelope_missing[index]
                            || merged_nullable,
                    ));
                } else {
                    target.envelope_types[index] = Some(with_nullable(
                        source_type.clone(),
                        target.envelope_missing[index]
                            || source.envelope_missing[index]
                            || source_type.nullable,
                    ));
                }
            }
            None => {
                target.envelope_missing[index] = true;
                if let Some(existing) = target.envelope_types[index].as_mut() {
                    existing.nullable = true;
                }
            }
        }
        target.envelope_missing[index] |= source.envelope_missing[index];
    }

    if let Some(source_payload) = source.payload_type.as_ref() {
        target.payload_type = Some(match target.payload_type.as_ref() {
            Some(existing) => merge_discovered_types(
                "payload",
                existing,
                source_payload,
                &options.widening_options,
            )?,
            None => source_payload.clone(),
        });
    }

    Ok(())
}

fn merge_discovered_types(
    path: &str,
    left: &DiscoveredType,
    right: &DiscoveredType,
    options: &WideningOptions,
) -> Result<DiscoveredType, String> {
    let nullable = left.nullable || right.nullable;

    match (&left.kind, &right.kind) {
        (DiscoveredKind::Null, _) => Ok(with_nullable(right.clone(), true)),
        (_, DiscoveredKind::Null) => Ok(with_nullable(left.clone(), true)),
        (DiscoveredKind::Struct(left_fields), DiscoveredKind::Struct(right_fields)) => {
            Ok(DiscoveredType {
                nullable,
                kind: DiscoveredKind::Struct(merge_discovered_fields(
                    path,
                    left_fields,
                    right_fields,
                    options,
                )?),
            })
        }
        (DiscoveredKind::List(left_element), DiscoveredKind::List(right_element)) => {
            Ok(DiscoveredType {
                nullable,
                kind: DiscoveredKind::List(Box::new(merge_discovered_types(
                    &format!("{path}[]"),
                    left_element,
                    right_element,
                    options,
                )?)),
            })
        }
        _ if is_discovered_primitive(left) && is_discovered_primitive(right) => {
            let widened = widen_data_type(
                &discovered_to_arrow_data_type(left),
                &discovered_to_arrow_data_type(right),
                options,
            )?;
            Ok(DiscoveredType {
                nullable,
                kind: discovered_kind_from_arrow(&widened)?,
            })
        }
        _ => Err(format!(
            "field `{}` incompatible types for widening: left={:?}, right={:?}",
            path,
            discovered_to_arrow_data_type(left),
            discovered_to_arrow_data_type(right)
        )),
    }
}

fn merge_discovered_fields(
    path: &str,
    left: &[DiscoveredField],
    right: &[DiscoveredField],
    options: &WideningOptions,
) -> Result<Vec<DiscoveredField>, String> {
    let right_by_name: HashMap<&str, &DiscoveredField> = right
        .iter()
        .map(|field| (field.name.as_str(), field))
        .collect();
    let left_names: BTreeSet<&str> = left.iter().map(|field| field.name.as_str()).collect();

    let mut merged = Vec::with_capacity(left.len() + right.len());
    for left_field in left {
        let child_path = if path.is_empty() {
            left_field.name.clone()
        } else {
            format!("{path}.{}", left_field.name)
        };

        if let Some(right_field) = right_by_name.get(left_field.name.as_str()) {
            merged.push(DiscoveredField {
                name: left_field.name.clone(),
                data_type: merge_discovered_types(
                    &child_path,
                    &left_field.data_type,
                    &right_field.data_type,
                    options,
                )?,
            });
        } else {
            merged.push(DiscoveredField {
                name: left_field.name.clone(),
                data_type: with_nullable(left_field.data_type.clone(), true),
            });
        }
    }

    for right_field in right {
        if !left_names.contains(right_field.name.as_str()) {
            merged.push(DiscoveredField {
                name: right_field.name.clone(),
                data_type: with_nullable(right_field.data_type.clone(), true),
            });
        }
    }

    merged.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    Ok(merged)
}

fn discovered_to_arrow_data_type(discovered: &DiscoveredType) -> DataType {
    match &discovered.kind {
        DiscoveredKind::Null => DataType::Null,
        DiscoveredKind::Boolean => DataType::Boolean,
        DiscoveredKind::Int64 => DataType::Int64,
        DiscoveredKind::UInt64 => DataType::UInt64,
        DiscoveredKind::Float64 => DataType::Float64,
        DiscoveredKind::Utf8 => DataType::Utf8,
        DiscoveredKind::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|field| {
                    Arc::new(Field::new(
                        &field.name,
                        discovered_to_arrow_data_type(&field.data_type),
                        field.data_type.nullable,
                    ))
                })
                .collect::<Vec<_>>()
                .into(),
        ),
        DiscoveredKind::List(element) => DataType::List(Arc::new(Field::new(
            "item",
            discovered_to_arrow_data_type(element),
            element.nullable,
        ))),
    }
}

fn discovered_kind_from_arrow(data_type: &DataType) -> Result<DiscoveredKind, String> {
    match data_type {
        DataType::Null => Ok(DiscoveredKind::Null),
        DataType::Boolean => Ok(DiscoveredKind::Boolean),
        DataType::Int64 => Ok(DiscoveredKind::Int64),
        DataType::UInt64 => Ok(DiscoveredKind::UInt64),
        DataType::Float64 => Ok(DiscoveredKind::Float64),
        DataType::Utf8 | DataType::LargeUtf8 => Ok(DiscoveredKind::Utf8),
        other => Err(format!(
            "unsupported widened discovery type generated from Arrow: {other:?}"
        )),
    }
}

fn with_nullable(mut discovered: DiscoveredType, nullable: bool) -> DiscoveredType {
    discovered.nullable |= nullable;
    discovered
}

fn is_discovered_primitive(discovered: &DiscoveredType) -> bool {
    matches!(
        discovered.kind,
        DiscoveredKind::Null
            | DiscoveredKind::Boolean
            | DiscoveredKind::Int64
            | DiscoveredKind::UInt64
            | DiscoveredKind::Float64
            | DiscoveredKind::Utf8
    )
}

fn compute_shape_key(value: &BorrowedValue<'_>, buffer: &mut String) -> ShapeKey {
    buffer.clear();
    push_shape_key(value, buffer);
    ShapeKey(buffer.clone())
}

fn push_shape_key(value: &BorrowedValue<'_>, output: &mut String) {
    if value.is_null() {
        output.push('n');
    } else if value.as_bool().is_some() {
        output.push('b');
    } else if value.as_i64().is_some() {
        output.push('i');
    } else if value.as_u64().is_some() {
        output.push('u');
    } else if value.as_f64().is_some() {
        output.push('f');
    } else if value.as_str().is_some() {
        output.push('s');
    } else if let Some(array) = value.as_array() {
        output.push('[');
        let mut entries = Vec::with_capacity(array.len());
        for entry in array.iter() {
            let mut nested = String::new();
            push_shape_key(entry, &mut nested);
            entries.push(nested);
        }
        entries.sort_unstable();
        entries.dedup();
        for entry in entries {
            output.push_str(&entry);
            output.push('|');
        }
        output.push(']');
    } else if let Some(object) = value.as_object() {
        output.push('{');
        let mut entries = Vec::with_capacity(object.len());
        for (key, value) in object.iter() {
            let mut nested = String::new();
            push_shape_key(value, &mut nested);
            entries.push((key.to_string(), nested));
        }
        entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        for (key, nested) in entries {
            output.push_str(&key);
            output.push(':');
            output.push_str(&nested);
            output.push(',');
        }
        output.push('}');
    } else {
        output.push('?');
    }
}

fn read_next_ndjson_value<'a>(
    reader: &mut BufReader<StdFile>,
    line_buffer: &'a mut Vec<u8>,
    parser_buffers: &mut Buffers,
) -> Result<Option<BorrowedValue<'a>>, String> {
    loop {
        line_buffer.clear();
        let bytes_read = reader
            .read_until(b'\n', line_buffer)
            .map_err(|error| error.to_string())?;
        if bytes_read == 0 {
            return Ok(None);
        }

        trim_line_endings(line_buffer);
        if line_buffer.is_empty() {
            continue;
        }

        return to_borrowed_value_with_buffers(line_buffer.as_mut_slice(), parser_buffers)
            .map(Some)
            .map_err(|error| error.to_string());
    }
}

fn trim_line_endings(buffer: &mut Vec<u8>) {
    while matches!(buffer.last(), Some(b'\n' | b'\r')) {
        buffer.pop();
    }
}

impl CompiledNdjsonPlan {
    fn compile(schema: &Schema, options: &CompactionOptions) -> Result<Self, String> {
        let payload_builder_index = schema.index_of(&options.payload_column).map_err(|_| {
            format!(
                "output schema is missing payload column `{}`",
                options.payload_column
            )
        })?;
        let payload_field = schema.field(payload_builder_index);
        let payload_fields = match payload_field.data_type() {
            DataType::Struct(fields) => fields,
            other => {
                return Err(format!(
                    "payload column `{}` must be Struct, found {other:?}",
                    options.payload_column
                ));
            }
        };

        let mut envelope_ops = Vec::new();
        for (index, field) in schema.fields().iter().enumerate() {
            if index == payload_builder_index {
                continue;
            }
            envelope_ops.push(CompiledNdjsonFieldOp {
                name: field.name().to_string(),
                builder_index: index,
                value_op: compile_ndjson_value_op(field.data_type())?,
            });
        }

        let payload_op = compile_ndjson_struct_op(payload_fields)?;
        let mut top_level_lookup = HashMap::new();
        for (index, field_op) in envelope_ops.iter().enumerate() {
            top_level_lookup.insert(field_op.name.clone(), RootDispatch::Envelope(index));
        }
        for (index, field_op) in payload_op.fields.iter().enumerate() {
            top_level_lookup.insert(field_op.name.clone(), RootDispatch::Payload(index));
        }

        Ok(Self {
            envelope_ops,
            payload_op,
            payload_builder_index,
            top_level_lookup,
        })
    }

    fn append_record(
        &self,
        root_builder: &mut NdjsonStructBuilder,
        record: &BorrowedValue<'_>,
        scratch: &mut NdjsonAppendScratch,
    ) -> Result<(), String> {
        let object = record
            .as_object()
            .ok_or_else(|| "NDJSON record must be an object".to_string())?;
        let mut seen = scratch.take_seen(0, self.envelope_ops.len() + self.payload_op.fields.len());
        let (envelope_seen, payload_seen) = seen.split_at_mut(self.envelope_ops.len());

        let (envelope_builders, payload_slice) = root_builder
            .children
            .split_at_mut(self.payload_builder_index);
        let payload_builder = payload_slice
            .first_mut()
            .ok_or_else(|| "root payload builder is missing".to_string())?
            .as_struct_mut()
            .ok_or_else(|| "root payload builder must be Struct".to_string())?;

        for (key, value) in object.iter() {
            match self.top_level_lookup.get(key.as_ref()) {
                Some(RootDispatch::Envelope(index)) => {
                    if !envelope_seen[*index] {
                        let op = &self.envelope_ops[*index];
                        op.append_to(
                            &mut envelope_builders[op.builder_index],
                            Some(value),
                            scratch,
                            1,
                        )?;
                        envelope_seen[*index] = true;
                    }
                }
                Some(RootDispatch::Payload(index)) => {
                    if !payload_seen[*index] {
                        let op = &self.payload_op.fields[*index];
                        op.append_to(
                            &mut payload_builder.children[op.builder_index],
                            Some(value),
                            scratch,
                            1,
                        )?;
                        payload_seen[*index] = true;
                    }
                }
                None => {}
            }
        }

        for (index, op) in self.envelope_ops.iter().enumerate() {
            if !envelope_seen[index] {
                op.append_to(&mut envelope_builders[op.builder_index], None, scratch, 1)?;
            }
        }
        for (index, op) in self.payload_op.fields.iter().enumerate() {
            if !payload_seen[index] {
                op.append_to(
                    &mut payload_builder.children[op.builder_index],
                    None,
                    scratch,
                    1,
                )?;
            }
        }

        payload_builder.append(true);
        root_builder.append(true);
        scratch.return_seen(0, seen);
        Ok(())
    }
}

impl CompiledNdjsonFieldOp {
    fn append_to(
        &self,
        builder: &mut NdjsonValueBuilder,
        value: Option<&BorrowedValue<'_>>,
        scratch: &mut NdjsonAppendScratch,
        depth: usize,
    ) -> Result<(), String> {
        self.value_op.append_to(builder, value, scratch, depth)
    }
}

impl CompiledNdjsonValueOp {
    fn append_to(
        &self,
        builder: &mut NdjsonValueBuilder,
        value: Option<&BorrowedValue<'_>>,
        scratch: &mut NdjsonAppendScratch,
        depth: usize,
    ) -> Result<(), String> {
        match self {
            Self::Null => {
                if value.is_some_and(|value| !value.is_null()) {
                    return Err(format!(
                        "expected null value for field, found {}",
                        borrowed_json_kind(value.expect("value exists"))
                    ));
                }
                builder.append_null_value()
            }
            Self::Boolean => match value {
                Some(value) if !value.is_null() => {
                    let boolean = value.as_bool().ok_or_else(|| {
                        format!(
                            "expected boolean for field, found {}",
                            borrowed_json_kind(value)
                        )
                    })?;
                    builder.append_boolean(boolean)
                }
                _ => builder.append_null_value(),
            },
            Self::Int8 => append_borrowed_signed(builder, value, "Int8"),
            Self::Int16 => append_borrowed_signed(builder, value, "Int16"),
            Self::Int32 => append_borrowed_signed(builder, value, "Int32"),
            Self::Int64 => append_borrowed_signed(builder, value, "Int64"),
            Self::UInt8 => append_borrowed_unsigned(builder, value, "UInt8"),
            Self::UInt16 => append_borrowed_unsigned(builder, value, "UInt16"),
            Self::UInt32 => append_borrowed_unsigned(builder, value, "UInt32"),
            Self::UInt64 => append_borrowed_unsigned(builder, value, "UInt64"),
            Self::Float32 => match value {
                Some(value) if !value.is_null() => {
                    let number = borrowed_numeric_to_f64(value).ok_or_else(|| {
                        format!(
                            "expected numeric value for Float32, found {}",
                            borrowed_json_kind(value)
                        )
                    })?;
                    builder.append_float32(number as f32)
                }
                _ => builder.append_null_value(),
            },
            Self::Float64 => match value {
                Some(value) if !value.is_null() => {
                    let number = borrowed_numeric_to_f64(value).ok_or_else(|| {
                        format!(
                            "expected numeric value for Float64, found {}",
                            borrowed_json_kind(value)
                        )
                    })?;
                    builder.append_float64(number)
                }
                _ => builder.append_null_value(),
            },
            Self::Utf8 | Self::LargeUtf8 => match value {
                Some(value) if !value.is_null() => {
                    let rendered = render_borrowed_string(value)?;
                    if matches!(self, Self::LargeUtf8) {
                        builder.append_large_string(&rendered)
                    } else {
                        builder.append_string(&rendered)
                    }
                }
                _ => builder.append_null_value(),
            },
            Self::Struct(struct_op) => struct_op.append_to(builder, value, scratch, depth),
            Self::List(element_op) => {
                append_borrowed_list(builder, value, element_op, scratch, depth)
            }
            Self::LargeList(element_op) => {
                append_borrowed_large_list(builder, value, element_op, scratch, depth)
            }
        }
    }
}

impl CompiledNdjsonStructOp {
    fn append_to(
        &self,
        builder: &mut NdjsonValueBuilder,
        value: Option<&BorrowedValue<'_>>,
        scratch: &mut NdjsonAppendScratch,
        depth: usize,
    ) -> Result<(), String> {
        let builder = builder
            .as_struct_mut()
            .ok_or_else(|| "compiled NDJSON struct op expected struct builder".to_string())?;

        match value {
            Some(value) if !value.is_null() => {
                let object = value.as_object().ok_or_else(|| {
                    format!(
                        "expected object for Struct field, found {}",
                        borrowed_json_kind(value)
                    )
                })?;
                let mut seen = scratch.take_seen(depth, self.fields.len());
                for (key, child_value) in object.iter() {
                    if let Some(index) = self.lookup.get(key.as_ref()) {
                        if !seen[*index] {
                            let op = &self.fields[*index];
                            op.append_to(
                                &mut builder.children[op.builder_index],
                                Some(child_value),
                                scratch,
                                depth + 1,
                            )?;
                            seen[*index] = true;
                        }
                    }
                }
                for (index, op) in self.fields.iter().enumerate() {
                    if !seen[index] {
                        op.append_to(
                            &mut builder.children[op.builder_index],
                            None,
                            scratch,
                            depth + 1,
                        )?;
                    }
                }
                builder.append(true);
                scratch.return_seen(depth, seen);
                Ok(())
            }
            _ => {
                for op in &self.fields {
                    op.append_to(
                        &mut builder.children[op.builder_index],
                        None,
                        scratch,
                        depth + 1,
                    )?;
                }
                builder.append(false);
                Ok(())
            }
        }
    }
}

impl NdjsonBatchBuilder {
    fn new(schema: SchemaRef, plan: CompiledNdjsonPlan, batch_rows: usize) -> Result<Self, String> {
        Ok(Self {
            batch_rows,
            plan,
            root: NdjsonStructBuilder::from_fields(schema.fields().clone(), batch_rows),
            scratch: NdjsonAppendScratch::default(),
        })
    }

    fn append_record(&mut self, record: &BorrowedValue<'_>) -> Result<(), String> {
        self.plan
            .append_record(&mut self.root, record, &mut self.scratch)
    }

    fn is_empty(&self) -> bool {
        self.root.len() == 0
    }

    fn is_full(&self) -> bool {
        self.root.len() >= self.batch_rows
    }

    fn finish_batch(&mut self) -> RecordBatch {
        RecordBatch::from(self.root.finish())
    }
}

impl NdjsonStructBuilder {
    fn from_fields(fields: Fields, capacity: usize) -> Self {
        let children = fields
            .iter()
            .map(|field| NdjsonValueBuilder::from_data_type(field.data_type(), capacity))
            .collect();
        Self {
            fields,
            children,
            nulls: NullBufferBuilder::new(capacity),
        }
    }

    fn len(&self) -> usize {
        self.nulls.len()
    }

    fn append(&mut self, is_valid: bool) {
        self.nulls.append(is_valid);
    }

    fn finish(&mut self) -> StructArray {
        let arrays = self
            .children
            .iter_mut()
            .map(|child| child.finish())
            .collect();
        StructArray::new(self.fields.clone(), arrays, self.nulls.finish())
    }

    fn finish_cloned(&self) -> StructArray {
        let arrays = self
            .children
            .iter()
            .map(NdjsonValueBuilder::finish_cloned)
            .collect();
        StructArray::new(self.fields.clone(), arrays, self.nulls.finish_cloned())
    }
}

impl NdjsonValueBuilder {
    fn from_data_type(data_type: &DataType, capacity: usize) -> Self {
        match data_type {
            DataType::Null => Self::Null(NullBuilder::new()),
            DataType::Boolean => Self::Boolean(BooleanBuilder::with_capacity(capacity)),
            DataType::Int8 => Self::Int8(Int8Builder::with_capacity(capacity)),
            DataType::Int16 => Self::Int16(Int16Builder::with_capacity(capacity)),
            DataType::Int32 => Self::Int32(Int32Builder::with_capacity(capacity)),
            DataType::Int64 => Self::Int64(Int64Builder::with_capacity(capacity)),
            DataType::UInt8 => Self::UInt8(UInt8Builder::with_capacity(capacity)),
            DataType::UInt16 => Self::UInt16(UInt16Builder::with_capacity(capacity)),
            DataType::UInt32 => Self::UInt32(UInt32Builder::with_capacity(capacity)),
            DataType::UInt64 => Self::UInt64(UInt64Builder::with_capacity(capacity)),
            DataType::Float32 => Self::Float32(Float32Builder::with_capacity(capacity)),
            DataType::Float64 => Self::Float64(Float64Builder::with_capacity(capacity)),
            DataType::Utf8 => Self::Utf8(StringBuilder::with_capacity(capacity, capacity * 16)),
            DataType::LargeUtf8 => {
                Self::LargeUtf8(LargeStringBuilder::with_capacity(capacity, capacity * 16))
            }
            DataType::Struct(fields) => {
                Self::Struct(NdjsonStructBuilder::from_fields(fields.clone(), capacity))
            }
            DataType::List(field) => Self::List(Box::new(
                ListBuilder::with_capacity(
                    NdjsonValueBuilder::from_data_type(field.data_type(), capacity),
                    capacity,
                )
                .with_field(field.clone()),
            )),
            DataType::LargeList(field) => Self::LargeList(Box::new(
                LargeListBuilder::with_capacity(
                    NdjsonValueBuilder::from_data_type(field.data_type(), capacity),
                    capacity,
                )
                .with_field(field.clone()),
            )),
            other => panic!("unsupported NDJSON builder type: {other:?}"),
        }
    }

    fn append_null_value(&mut self) -> Result<(), String> {
        match self {
            Self::Null(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::Boolean(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::Int8(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::Int16(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::Int32(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::Int64(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::UInt8(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::UInt16(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::UInt32(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::UInt64(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::Float32(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::Float64(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::Utf8(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::LargeUtf8(builder) => {
                builder.append_null();
                Ok(())
            }
            Self::Struct(builder) => {
                for child in &mut builder.children {
                    child.append_null_value()?;
                }
                builder.append(false);
                Ok(())
            }
            Self::List(builder) => {
                builder.append(false);
                Ok(())
            }
            Self::LargeList(builder) => {
                builder.append(false);
                Ok(())
            }
        }
    }

    fn append_boolean(&mut self, value: bool) -> Result<(), String> {
        match self {
            Self::Boolean(builder) => {
                builder.append_value(value);
                Ok(())
            }
            _ => Err("expected boolean builder".to_string()),
        }
    }

    fn append_i64(&mut self, type_name: &str, value: i64) -> Result<(), String> {
        match (self, type_name) {
            (Self::Int8(builder), "Int8") => {
                builder.append_value(
                    i8::try_from(value)
                        .map_err(|_| format!("value `{value}` is out of range for Int8"))?,
                );
                Ok(())
            }
            (Self::Int16(builder), "Int16") => {
                builder.append_value(
                    i16::try_from(value)
                        .map_err(|_| format!("value `{value}` is out of range for Int16"))?,
                );
                Ok(())
            }
            (Self::Int32(builder), "Int32") => {
                builder.append_value(
                    i32::try_from(value)
                        .map_err(|_| format!("value `{value}` is out of range for Int32"))?,
                );
                Ok(())
            }
            (Self::Int64(builder), "Int64") => {
                builder.append_value(value);
                Ok(())
            }
            _ => Err(format!("expected signed builder for {type_name}")),
        }
    }

    fn append_u64(&mut self, type_name: &str, value: u64) -> Result<(), String> {
        match (self, type_name) {
            (Self::UInt8(builder), "UInt8") => {
                builder.append_value(
                    u8::try_from(value)
                        .map_err(|_| format!("value `{value}` is out of range for UInt8"))?,
                );
                Ok(())
            }
            (Self::UInt16(builder), "UInt16") => {
                builder.append_value(
                    u16::try_from(value)
                        .map_err(|_| format!("value `{value}` is out of range for UInt16"))?,
                );
                Ok(())
            }
            (Self::UInt32(builder), "UInt32") => {
                builder.append_value(
                    u32::try_from(value)
                        .map_err(|_| format!("value `{value}` is out of range for UInt32"))?,
                );
                Ok(())
            }
            (Self::UInt64(builder), "UInt64") => {
                builder.append_value(value);
                Ok(())
            }
            _ => Err(format!("expected unsigned builder for {type_name}")),
        }
    }

    fn append_float32(&mut self, value: f32) -> Result<(), String> {
        match self {
            Self::Float32(builder) => {
                builder.append_value(value);
                Ok(())
            }
            _ => Err("expected Float32 builder".to_string()),
        }
    }

    fn append_float64(&mut self, value: f64) -> Result<(), String> {
        match self {
            Self::Float64(builder) => {
                builder.append_value(value);
                Ok(())
            }
            _ => Err("expected Float64 builder".to_string()),
        }
    }

    fn append_string(&mut self, value: &str) -> Result<(), String> {
        match self {
            Self::Utf8(builder) => {
                builder.append_value(value);
                Ok(())
            }
            _ => Err("expected Utf8 builder".to_string()),
        }
    }

    fn append_large_string(&mut self, value: &str) -> Result<(), String> {
        match self {
            Self::LargeUtf8(builder) => {
                builder.append_value(value);
                Ok(())
            }
            _ => Err("expected LargeUtf8 builder".to_string()),
        }
    }

    fn as_struct_mut(&mut self) -> Option<&mut NdjsonStructBuilder> {
        match self {
            Self::Struct(builder) => Some(builder),
            _ => None,
        }
    }

    fn as_list_mut(&mut self) -> Option<&mut ListBuilder<NdjsonValueBuilder>> {
        match self {
            Self::List(builder) => Some(builder.as_mut()),
            _ => None,
        }
    }

    fn as_large_list_mut(&mut self) -> Option<&mut LargeListBuilder<NdjsonValueBuilder>> {
        match self {
            Self::LargeList(builder) => Some(builder.as_mut()),
            _ => None,
        }
    }
}

impl ArrayBuilder for NdjsonValueBuilder {
    fn len(&self) -> usize {
        match self {
            Self::Null(builder) => builder.len(),
            Self::Boolean(builder) => builder.len(),
            Self::Int8(builder) => builder.len(),
            Self::Int16(builder) => builder.len(),
            Self::Int32(builder) => builder.len(),
            Self::Int64(builder) => builder.len(),
            Self::UInt8(builder) => builder.len(),
            Self::UInt16(builder) => builder.len(),
            Self::UInt32(builder) => builder.len(),
            Self::UInt64(builder) => builder.len(),
            Self::Float32(builder) => builder.len(),
            Self::Float64(builder) => builder.len(),
            Self::Utf8(builder) => builder.len(),
            Self::LargeUtf8(builder) => builder.len(),
            Self::Struct(builder) => builder.len(),
            Self::List(builder) => builder.len(),
            Self::LargeList(builder) => builder.len(),
        }
    }

    fn finish(&mut self) -> ArrayRef {
        match self {
            Self::Null(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Boolean(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Int8(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Int16(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Int32(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Int64(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::UInt8(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::UInt16(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::UInt32(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::UInt64(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Float32(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Float64(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Utf8(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::LargeUtf8(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::Struct(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::List(builder) => Arc::new(builder.finish()) as ArrayRef,
            Self::LargeList(builder) => Arc::new(builder.finish()) as ArrayRef,
        }
    }

    fn finish_cloned(&self) -> ArrayRef {
        match self {
            Self::Null(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::Boolean(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::Int8(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::Int16(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::Int32(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::Int64(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::UInt8(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::UInt16(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::UInt32(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::UInt64(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::Float32(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::Float64(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::Utf8(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::LargeUtf8(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::Struct(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::List(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
            Self::LargeList(builder) => Arc::new(builder.finish_cloned()) as ArrayRef,
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

fn compile_ndjson_struct_op(fields: &Fields) -> Result<CompiledNdjsonStructOp, String> {
    let mut field_ops = Vec::with_capacity(fields.len());
    let mut lookup = HashMap::with_capacity(fields.len());
    for (index, field) in fields.iter().enumerate() {
        let field_op = CompiledNdjsonFieldOp {
            name: field.name().to_string(),
            builder_index: index,
            value_op: compile_ndjson_value_op(field.data_type())?,
        };
        lookup.insert(field_op.name.clone(), index);
        field_ops.push(field_op);
    }
    Ok(CompiledNdjsonStructOp {
        fields: field_ops,
        lookup,
    })
}

fn compile_ndjson_value_op(data_type: &DataType) -> Result<CompiledNdjsonValueOp, String> {
    match data_type {
        DataType::Null => Ok(CompiledNdjsonValueOp::Null),
        DataType::Boolean => Ok(CompiledNdjsonValueOp::Boolean),
        DataType::Int8 => Ok(CompiledNdjsonValueOp::Int8),
        DataType::Int16 => Ok(CompiledNdjsonValueOp::Int16),
        DataType::Int32 => Ok(CompiledNdjsonValueOp::Int32),
        DataType::Int64 => Ok(CompiledNdjsonValueOp::Int64),
        DataType::UInt8 => Ok(CompiledNdjsonValueOp::UInt8),
        DataType::UInt16 => Ok(CompiledNdjsonValueOp::UInt16),
        DataType::UInt32 => Ok(CompiledNdjsonValueOp::UInt32),
        DataType::UInt64 => Ok(CompiledNdjsonValueOp::UInt64),
        DataType::Float32 => Ok(CompiledNdjsonValueOp::Float32),
        DataType::Float64 => Ok(CompiledNdjsonValueOp::Float64),
        DataType::Utf8 => Ok(CompiledNdjsonValueOp::Utf8),
        DataType::LargeUtf8 => Ok(CompiledNdjsonValueOp::LargeUtf8),
        DataType::Struct(fields) => Ok(CompiledNdjsonValueOp::Struct(compile_ndjson_struct_op(
            fields,
        )?)),
        DataType::List(field) => Ok(CompiledNdjsonValueOp::List(Box::new(
            compile_ndjson_value_op(field.data_type())?,
        ))),
        DataType::LargeList(field) => Ok(CompiledNdjsonValueOp::LargeList(Box::new(
            compile_ndjson_value_op(field.data_type())?,
        ))),
        other => Err(format!(
            "NDJSON compaction does not yet support target type {other:?}"
        )),
    }
}

fn append_borrowed_signed(
    builder: &mut NdjsonValueBuilder,
    value: Option<&BorrowedValue<'_>>,
    type_name: &str,
) -> Result<(), String> {
    match value {
        Some(value) if !value.is_null() => {
            let number = value.as_i64().ok_or_else(|| {
                format!(
                    "expected integer value for {type_name}, found {}",
                    borrowed_json_kind(value)
                )
            })?;
            builder.append_i64(type_name, number)
        }
        _ => builder.append_null_value(),
    }
}

fn append_borrowed_unsigned(
    builder: &mut NdjsonValueBuilder,
    value: Option<&BorrowedValue<'_>>,
    type_name: &str,
) -> Result<(), String> {
    match value {
        Some(value) if !value.is_null() => {
            let number = value.as_u64().ok_or_else(|| {
                format!(
                    "expected unsigned integer value for {type_name}, found {}",
                    borrowed_json_kind(value)
                )
            })?;
            builder.append_u64(type_name, number)
        }
        _ => builder.append_null_value(),
    }
}

fn append_borrowed_list(
    builder: &mut NdjsonValueBuilder,
    value: Option<&BorrowedValue<'_>>,
    element_op: &CompiledNdjsonValueOp,
    scratch: &mut NdjsonAppendScratch,
    depth: usize,
) -> Result<(), String> {
    match value {
        Some(value) if !value.is_null() => {
            let values = value.as_array().ok_or_else(|| {
                format!(
                    "expected array for List field, found {}",
                    borrowed_json_kind(value)
                )
            })?;
            let builder = builder
                .as_list_mut()
                .ok_or_else(|| "compiled NDJSON list op expected List builder".to_string())?;
            for value in values.iter() {
                element_op.append_to(builder.values(), Some(value), scratch, depth + 1)?;
            }
            builder.append(true);
            Ok(())
        }
        _ => builder.append_null_value(),
    }
}

fn append_borrowed_large_list(
    builder: &mut NdjsonValueBuilder,
    value: Option<&BorrowedValue<'_>>,
    element_op: &CompiledNdjsonValueOp,
    scratch: &mut NdjsonAppendScratch,
    depth: usize,
) -> Result<(), String> {
    match value {
        Some(value) if !value.is_null() => {
            let values = value.as_array().ok_or_else(|| {
                format!(
                    "expected array for LargeList field, found {}",
                    borrowed_json_kind(value)
                )
            })?;
            let builder = builder.as_large_list_mut().ok_or_else(|| {
                "compiled NDJSON large-list op expected LargeList builder".to_string()
            })?;
            for value in values.iter() {
                element_op.append_to(builder.values(), Some(value), scratch, depth + 1)?;
            }
            builder.append(true);
            Ok(())
        }
        _ => builder.append_null_value(),
    }
}

fn render_borrowed_string(value: &BorrowedValue<'_>) -> Result<String, String> {
    if let Some(string) = value.as_str() {
        return Ok(string.to_owned());
    }
    if let Some(boolean) = value.as_bool() {
        return Ok(boolean.to_string());
    }
    if let Some(number) = value.as_i64() {
        return Ok(number.to_string());
    }
    if let Some(number) = value.as_u64() {
        return Ok(number.to_string());
    }
    if let Some(number) = value.as_f64() {
        return Ok(number.to_string());
    }
    Err(format!(
        "expected primitive or string value for string field, found {}",
        borrowed_json_kind(value)
    ))
}

fn borrowed_numeric_to_f64(value: &BorrowedValue<'_>) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|number| number as f64))
        .or_else(|| value.as_u64().map(|number| number as f64))
}

fn borrowed_json_kind(value: &BorrowedValue<'_>) -> &'static str {
    if value.is_null() {
        "null"
    } else if value.as_bool().is_some() {
        "bool"
    } else if value.as_i64().is_some() || value.as_u64().is_some() || value.as_f64().is_some() {
        "number"
    } else if value.as_str().is_some() {
        "string"
    } else if value.as_array().is_some() {
        "array"
    } else if value.as_object().is_some() {
        "object"
    } else {
        "unknown"
    }
}

fn merge_payload_schemas_many<'a, I>(
    schemas: I,
    options: &PayloadMergeOptions,
) -> Result<Schema, String>
where
    I: IntoIterator<Item = &'a Schema>,
{
    let mut schemas = schemas.into_iter();
    let first = schemas
        .next()
        .ok_or_else(|| "at least one schema is required to build a payload plan".to_string())?;
    let mut merged = first.clone();

    for schema in schemas {
        merged = merge_payload_schemas_pair(&merged, schema, options)?;
    }

    Ok(merged)
}

fn compile_type_adapter(
    source_type: &DataType,
    target_type: &DataType,
) -> Result<CompiledTypeAdapter, String> {
    if source_type == target_type {
        return Ok(CompiledTypeAdapter::PassThrough);
    }

    if is_primitive_or_string(source_type) && is_primitive_or_string(target_type) {
        return Ok(CompiledTypeAdapter::PrimitiveCast {
            target_type: target_type.clone(),
        });
    }

    match (source_type, target_type) {
        (DataType::Struct(source_fields), DataType::Struct(target_fields)) => {
            let source_lookup: HashMap<&str, usize> = source_fields
                .iter()
                .enumerate()
                .map(|(index, field)| (field.name().as_str(), index))
                .collect();
            let mut children = Vec::with_capacity(target_fields.len());
            for target_field in target_fields {
                let source_index = source_lookup.get(target_field.name().as_str()).copied();
                let adapter = match source_index {
                    Some(index) => compile_type_adapter(
                        source_fields[index].data_type(),
                        target_field.data_type(),
                    )?,
                    None => CompiledTypeAdapter::NullFill {
                        target_type: target_field.data_type().clone(),
                    },
                };
                children.push(CompiledStructChild {
                    source_index,
                    target_field: target_field.clone(),
                    adapter,
                });
            }

            Ok(CompiledTypeAdapter::StructAdapter(CompiledStructAdapter {
                target_fields: target_fields.clone(),
                children,
            }))
        }
        (DataType::List(source_field), DataType::List(target_field)) => {
            Ok(CompiledTypeAdapter::ListAdapter(CompiledListAdapter {
                target_type: target_type.clone(),
                target_field: target_field.clone(),
                value_adapter: Box::new(compile_type_adapter(
                    source_field.data_type(),
                    target_field.data_type(),
                )?),
                source_kind: ListKind::List,
            }))
        }
        (DataType::LargeList(source_field), DataType::List(target_field)) => {
            Ok(CompiledTypeAdapter::ListAdapter(CompiledListAdapter {
                target_type: target_type.clone(),
                target_field: target_field.clone(),
                value_adapter: Box::new(compile_type_adapter(
                    source_field.data_type(),
                    target_field.data_type(),
                )?),
                source_kind: ListKind::LargeList,
            }))
        }
        (DataType::List(source_field), DataType::LargeList(target_field)) => {
            Ok(CompiledTypeAdapter::LargeListAdapter(CompiledListAdapter {
                target_type: target_type.clone(),
                target_field: target_field.clone(),
                value_adapter: Box::new(compile_type_adapter(
                    source_field.data_type(),
                    target_field.data_type(),
                )?),
                source_kind: ListKind::List,
            }))
        }
        (DataType::LargeList(source_field), DataType::LargeList(target_field)) => {
            Ok(CompiledTypeAdapter::LargeListAdapter(CompiledListAdapter {
                target_type: target_type.clone(),
                target_field: target_field.clone(),
                value_adapter: Box::new(compile_type_adapter(
                    source_field.data_type(),
                    target_field.data_type(),
                )?),
                source_kind: ListKind::LargeList,
            }))
        }
        _ => Err(format!(
            "unsupported coercion from {source_type:?} to {target_type:?}"
        )),
    }
}

fn validate_compaction_options(options: &CompactionOptions) -> Result<(), String> {
    if options.payload_column.trim().is_empty() {
        return Err("payload_column must not be empty".to_string());
    }
    if options.batch_rows == 0 {
        return Err("batch_rows must be greater than zero".to_string());
    }

    let mut seen = BTreeSet::new();
    for field in &options.envelope_fields {
        if field == &options.payload_column {
            return Err(format!(
                "envelope field `{field}` conflicts with payload column `{}`",
                options.payload_column
            ));
        }
        if !seen.insert(field.clone()) {
            return Err(format!("duplicate envelope field `{field}`"));
        }
    }

    Ok(())
}

fn default_scan_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

fn schema_fingerprint(schema: &Schema) -> String {
    let mut parts = Vec::with_capacity(schema.fields().len() + schema.metadata().len());
    for field in schema.fields() {
        parts.push(field_fingerprint(field));
    }

    if !schema.metadata().is_empty() {
        let mut entries: Vec<_> = schema.metadata().iter().collect();
        entries.sort_unstable_by(|left, right| left.0.cmp(right.0).then(left.1.cmp(right.1)));
        parts.push(format!(
            "schema_meta({})",
            entries
                .into_iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }

    parts.join("|")
}

fn field_fingerprint(field: &Field) -> String {
    let mut rendered = format!(
        "{}:{}:{}",
        field.name(),
        field.is_nullable(),
        data_type_fingerprint(field.data_type())
    );

    if !field.metadata().is_empty() {
        let mut entries: Vec<_> = field.metadata().iter().collect();
        entries.sort_unstable_by(|left, right| left.0.cmp(right.0).then(left.1.cmp(right.1)));
        rendered.push('{');
        rendered.push_str(
            &entries
                .into_iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        rendered.push('}');
    }

    rendered
}

fn data_type_fingerprint(data_type: &DataType) -> String {
    match data_type {
        DataType::Struct(fields) => format!(
            "Struct({})",
            fields
                .iter()
                .map(|field| field_fingerprint(field))
                .collect::<Vec<_>>()
                .join(",")
        ),
        DataType::List(field) => format!("List({})", field_fingerprint(field)),
        DataType::LargeList(field) => format!("LargeList({})", field_fingerprint(field)),
        other => format!("{other:?}"),
    }
}

fn io_error(message: impl Into<String>) -> std::io::Error {
    std::io::Error::other(message.into())
}

fn peak_rss_bytes() -> u64 {
    unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return 0;
        }

        #[cfg(target_os = "macos")]
        {
            usage.ru_maxrss as u64
        }

        #[cfg(not(target_os = "macos"))]
        {
            (usage.ru_maxrss as u64) * 1024
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow_array::{Float64Array, Int32Array, StringArray};
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
        }
    }

    fn compaction_options_with_parallelism(parallelism: usize) -> CompactionOptions {
        let mut options = compaction_options();
        options.scan_parallelism = Some(parallelism);
        options
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
        let left_scores =
            ListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(vec![
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

    async fn write_parquet(
        path: &Path,
        schema: SchemaRef,
        batch: RecordBatch,
    ) -> Result<(), Box<dyn Error>> {
        let file = File::create(path).await?;
        let mut writer = AsyncArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch).await?;
        writer.close().await?;
        Ok(())
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

        let CompiledTypeAdapter::StructAdapter(profile_adapter) =
            &payload_adapter.children[1].adapter
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
            br#"{"event_id":2,"org_id":20,"score":4.5,"amount":9,"profile":{"tier":"gold"}}"#
                .to_vec();
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
}
