use std::cmp::Ordering as CmpOrdering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};
use std::error::Error;
use std::fs::File as StdFile;
use std::hash::{Hash, Hasher};
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
use arrow::compute::{cast, concat_batches, take_record_batch};
use arrow_array::builder::{
    ArrayBuilder, BooleanBuilder, Float32Builder, Float64Builder, Int8Builder, Int16Builder,
    Int32Builder, Int64Builder, LargeListBuilder, LargeStringBuilder, ListBuilder, NullBuilder,
    StringBuilder, UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
};
use arrow_array::{
    Array, ArrayRef, Date32Array, Date64Array, Float64Array, Int64Array, LargeListArray,
    LargeStringArray, ListArray, RecordBatch, StringArray, StructArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
    UInt64Array,
};
use arrow_buffer::NullBufferBuilder;
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use futures_util::StreamExt;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_writer::AsyncArrowWriter;
use parquet::file::properties::WriterProperties;
use simd_json::borrowed::Value as BorrowedValue;
use simd_json::prelude::{TypedScalarValue, ValueAsArray, ValueAsObject, ValueAsScalar};
use simd_json::{Buffers, to_borrowed_value_with_buffers};
use tokio::fs::File;
use tokio::io::BufWriter;
use tokio::sync::mpsc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionOptions {
    pub envelope_fields: Vec<String>,
    pub payload_column: String,
    pub widening_options: WideningOptions,
    pub batch_rows: usize,
    pub scan_parallelism: Option<usize>,
    pub read_buffer_bytes: usize,
    pub sort_field: Option<String>,
    pub sort_max_rows_in_memory: usize,
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
            sort_field: None,
            sort_max_rows_in_memory: 262_144,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParquetMergeExecutionOptions {
    pub ordering_field: Option<String>,
    pub read_batch_size: usize,
    pub output_batch_rows: usize,
    pub prefetch_batches_per_source: usize,
    pub output_row_group_rows: usize,
}

impl Default for ParquetMergeExecutionOptions {
    fn default() -> Self {
        Self {
            ordering_field: None,
            read_batch_size: 32_768,
            output_batch_rows: 32_768,
            prefetch_batches_per_source: 1,
            output_row_group_rows: 128_000,
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
    pub sorting_duration: Duration,
    pub total_duration: Duration,
    pub peak_rss_bytes: u64,
    pub planning_threads_used: usize,
    pub planning_unique_shapes: u64,
    pub planning_shape_cache_hits: u64,
    pub planning_shape_cache_misses: u64,
    pub input_batches: u64,
    pub output_batches: u64,
    pub adapter_cache_hits: u64,
    pub adapter_cache_misses: u64,
    pub ordered_merge_duration: Duration,
}

type InternedFieldName = Arc<str>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ShapeSignature(u64);

#[derive(Debug, Default)]
struct FieldInterner {
    names: HashMap<String, InternedFieldName>,
}

impl FieldInterner {
    fn intern(&mut self, name: &str) -> InternedFieldName {
        if let Some(existing) = self.names.get(name) {
            return existing.clone();
        }

        let owned = name.to_string();
        let interned: InternedFieldName = Arc::from(owned.as_str());
        self.names.insert(owned, interned.clone());
        interned
    }
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

#[derive(Clone, Debug)]
struct ParquetOrderPlan {
    field_name: String,
    field_index: usize,
    key_type: ParquetOrderKeyType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParquetOrderKeyType {
    Int64,
    UInt64,
    Float64,
    Utf8,
    LargeUtf8,
    Date32,
    Date64,
    TimestampSecond,
    TimestampMillisecond,
    TimestampMicrosecond,
    TimestampNanosecond,
}

#[derive(Clone, Debug)]
struct PreparedOrderBatch {
    batch: RecordBatch,
    order_column: PreparedOrderColumn,
}

#[derive(Clone, Debug)]
enum PreparedOrderColumn {
    Int64(Int64Array),
    UInt64(UInt64Array),
    Float64(Float64Array),
    Utf8(StringArray),
    LargeUtf8(LargeStringArray),
    Date32(Date32Array),
    Date64(Date64Array),
    TimestampSecond(TimestampSecondArray),
    TimestampMillisecond(TimestampMillisecondArray),
    TimestampMicrosecond(TimestampMicrosecondArray),
    TimestampNanosecond(TimestampNanosecondArray),
}

#[derive(Clone, Debug, PartialEq)]
enum OrderKeyValue {
    Int64(Option<i64>),
    UInt64(Option<u64>),
    Float64(Option<f64>),
    Utf8(Option<String>),
    LargeUtf8(Option<String>),
    Date32(Option<i32>),
    Date64(Option<i64>),
    Timestamp(Option<i64>),
}

#[derive(Clone, Debug, PartialEq)]
struct HeapEntry {
    source_index: usize,
    key: OrderKeyValue,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        compare_order_key_values(&other.key, &self.key)
            .then_with(|| other.source_index.cmp(&self.source_index))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct OrderedMergeSource {
    source_index: usize,
    receiver: mpsc::Receiver<Result<PreparedOrderBatch, String>>,
    current_batch: Option<PreparedOrderBatch>,
    current_row: usize,
}

impl OrderedMergeSource {
    fn new(
        source_index: usize,
        receiver: mpsc::Receiver<Result<PreparedOrderBatch, String>>,
    ) -> Self {
        Self {
            source_index,
            receiver,
            current_batch: None,
            current_row: 0,
        }
    }

    async fn load_next(&mut self, input_batches: &mut u64) -> Result<bool, String> {
        match self.receiver.recv().await {
            Some(Ok(batch)) => {
                *input_batches += 1;
                self.current_batch = Some(batch);
                self.current_row = 0;
                Ok(true)
            }
            Some(Err(error)) => Err(error),
            None => {
                self.current_batch = None;
                self.current_row = 0;
                Ok(false)
            }
        }
    }

    fn current_heap_entry(&self) -> Option<HeapEntry> {
        let batch = self.current_batch.as_ref()?;
        Some(HeapEntry {
            source_index: self.source_index,
            key: batch.order_column.key_at(self.current_row),
        })
    }

    fn contiguous_run_len(&self, next_competitor: Option<&HeapEntry>, max_rows: usize) -> usize {
        let Some(batch) = self.current_batch.as_ref() else {
            return 0;
        };
        if max_rows == 0 {
            return 0;
        }

        let available = batch.batch.num_rows() - self.current_row;
        let limit = available.min(max_rows);
        if limit == 0 {
            return 0;
        }

        let Some(competitor) = next_competitor else {
            return limit;
        };

        let mut run_len = 1;
        while run_len < limit {
            let row = self.current_row + run_len;
            if compare_row_against_heap_entry(
                &batch.order_column,
                row,
                competitor,
                self.source_index,
            ) == CmpOrdering::Greater
            {
                break;
            }
            run_len += 1;
        }
        run_len
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
    merge_payload_parquet_files_with_execution(
        input_paths,
        output_path,
        options,
        &ParquetMergeExecutionOptions::default(),
    )
    .await
}

pub async fn merge_payload_parquet_files_with_execution(
    input_paths: &[PathBuf],
    output_path: &Path,
    options: &PayloadMergeOptions,
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<CompactionReport, Box<dyn Error>> {
    if input_paths.is_empty() {
        return Err(io_error("at least one parquet input path is required").into());
    }
    validate_parquet_merge_execution_options(execution_options).map_err(io_error)?;

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
    let order_plan =
        build_parquet_order_plan(plan.output_schema.as_ref(), options, execution_options)
            .map_err(io_error)?;

    let mut source_adapters = Vec::with_capacity(schemas.len());
    let mut adapter_cache_hits = 0_u64;
    let mut adapter_cache_misses = 0_u64;
    let mut seen_fingerprints = HashSet::with_capacity(schemas.len());
    for schema in &schemas {
        let fingerprint = schema_fingerprint(schema.as_ref());
        if seen_fingerprints.insert(fingerprint) {
            adapter_cache_misses += 1;
        } else {
            adapter_cache_hits += 1;
        }
        source_adapters.push(plan.source_adapter_for_schema(schema.as_ref())?);
    }

    let planning_duration = planning_start.elapsed();
    let (rows, input_batches, output_batches, execution_duration, ordered_merge_duration) =
        if let Some(order_plan) = order_plan.as_ref() {
            merge_payload_parquet_files_ordered(
                input_paths,
                output_path,
                plan.output_schema.clone(),
                source_adapters,
                order_plan.clone(),
                execution_options,
            )
            .await
            .map_err(io_error)?
        } else {
            merge_payload_parquet_files_unordered(builders, output_path, plan, execution_options)
                .await
                .map_err(io_error)?
        };

    let output_bytes = std::fs::metadata(output_path)?.len();

    Ok(CompactionReport {
        rows,
        input_bytes,
        output_bytes,
        planning_duration,
        execution_duration,
        sorting_duration: Duration::default(),
        total_duration: total_start.elapsed(),
        peak_rss_bytes: peak_rss_bytes(),
        planning_threads_used: 0,
        planning_unique_shapes: 0,
        planning_shape_cache_hits: 0,
        planning_shape_cache_misses: 0,
        input_batches,
        output_batches,
        adapter_cache_hits,
        adapter_cache_misses,
        ordered_merge_duration,
    })
}

fn validate_parquet_merge_execution_options(
    options: &ParquetMergeExecutionOptions,
) -> Result<(), String> {
    if options.read_batch_size == 0 {
        return Err("read_batch_size must be greater than zero".to_string());
    }
    if options.output_batch_rows == 0 {
        return Err("output_batch_rows must be greater than zero".to_string());
    }
    if options.prefetch_batches_per_source == 0 {
        return Err("prefetch_batches_per_source must be greater than zero".to_string());
    }
    if options.output_row_group_rows == 0 {
        return Err("output_row_group_rows must be greater than zero".to_string());
    }
    if let Some(ordering_field) = options.ordering_field.as_ref() {
        if ordering_field.trim().is_empty() {
            return Err("ordering_field must not be empty when provided".to_string());
        }
    }
    Ok(())
}

fn build_parquet_order_plan(
    schema: &Schema,
    payload_options: &PayloadMergeOptions,
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<Option<ParquetOrderPlan>, String> {
    let Some(ordering_field) = execution_options.ordering_field.as_ref() else {
        return Ok(None);
    };

    if ordering_field == &payload_options.payload_column {
        return Err(format!(
            "ordering_field `{ordering_field}` must reference a top-level envelope column, not `{}`",
            payload_options.payload_column
        ));
    }

    let field_index = schema.index_of(ordering_field).map_err(|_| {
        format!("ordering_field `{ordering_field}` is missing from the merged schema")
    })?;
    let field = schema.field(field_index);
    let key_type = match field.data_type() {
        DataType::Int64 => ParquetOrderKeyType::Int64,
        DataType::UInt64 => ParquetOrderKeyType::UInt64,
        DataType::Float64 => ParquetOrderKeyType::Float64,
        DataType::Utf8 => ParquetOrderKeyType::Utf8,
        DataType::LargeUtf8 => ParquetOrderKeyType::LargeUtf8,
        DataType::Date32 => ParquetOrderKeyType::Date32,
        DataType::Date64 => ParquetOrderKeyType::Date64,
        DataType::Timestamp(time_unit, _) => match time_unit {
            arrow_schema::TimeUnit::Second => ParquetOrderKeyType::TimestampSecond,
            arrow_schema::TimeUnit::Millisecond => ParquetOrderKeyType::TimestampMillisecond,
            arrow_schema::TimeUnit::Microsecond => ParquetOrderKeyType::TimestampMicrosecond,
            arrow_schema::TimeUnit::Nanosecond => ParquetOrderKeyType::TimestampNanosecond,
        },
        other => {
            return Err(format!(
                "ordering_field `{ordering_field}` has unsupported type {other:?}; supported types are Int64, UInt64, Float64, Utf8, LargeUtf8, Date32, Date64, and Timestamp(_, _)"
            ));
        }
    };

    Ok(Some(ParquetOrderPlan {
        field_name: ordering_field.clone(),
        field_index,
        key_type,
    }))
}

fn parquet_writer_properties(execution_options: &ParquetMergeExecutionOptions) -> WriterProperties {
    WriterProperties::builder()
        .set_max_row_group_size(execution_options.output_row_group_rows)
        .set_write_batch_size(execution_options.output_batch_rows)
        .build()
}

fn materialize_pending_output(
    schema: &SchemaRef,
    pending_batches: &mut Vec<RecordBatch>,
) -> Result<Option<RecordBatch>, String> {
    if pending_batches.is_empty() {
        return Ok(None);
    }

    if pending_batches.len() == 1 {
        return Ok(Some(pending_batches.pop().expect("single pending batch")));
    }

    let combined = concat_batches(schema, pending_batches.iter())
        .map_err(|error| format!("failed to concatenate ordered merge output batches: {error}"))?;
    pending_batches.clear();
    Ok(Some(combined))
}

fn extract_prepared_order_column(
    batch: &RecordBatch,
    order_plan: &ParquetOrderPlan,
) -> Result<PreparedOrderColumn, String> {
    let column = batch.column(order_plan.field_index);
    match order_plan.key_type {
        ParquetOrderKeyType::Int64 => column
            .as_any()
            .downcast_ref::<Int64Array>()
            .cloned()
            .map(PreparedOrderColumn::Int64)
            .ok_or_else(|| {
                format!(
                    "ordering_field `{}` is not Int64 at runtime",
                    order_plan.field_name
                )
            }),
        ParquetOrderKeyType::UInt64 => column
            .as_any()
            .downcast_ref::<UInt64Array>()
            .cloned()
            .map(PreparedOrderColumn::UInt64)
            .ok_or_else(|| {
                format!(
                    "ordering_field `{}` is not UInt64 at runtime",
                    order_plan.field_name
                )
            }),
        ParquetOrderKeyType::Float64 => column
            .as_any()
            .downcast_ref::<Float64Array>()
            .cloned()
            .map(PreparedOrderColumn::Float64)
            .ok_or_else(|| {
                format!(
                    "ordering_field `{}` is not Float64 at runtime",
                    order_plan.field_name
                )
            }),
        ParquetOrderKeyType::Utf8 => column
            .as_any()
            .downcast_ref::<StringArray>()
            .cloned()
            .map(PreparedOrderColumn::Utf8)
            .ok_or_else(|| {
                format!(
                    "ordering_field `{}` is not Utf8 at runtime",
                    order_plan.field_name
                )
            }),
        ParquetOrderKeyType::LargeUtf8 => column
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .cloned()
            .map(PreparedOrderColumn::LargeUtf8)
            .ok_or_else(|| {
                format!(
                    "ordering_field `{}` is not LargeUtf8 at runtime",
                    order_plan.field_name
                )
            }),
        ParquetOrderKeyType::Date32 => column
            .as_any()
            .downcast_ref::<Date32Array>()
            .cloned()
            .map(PreparedOrderColumn::Date32)
            .ok_or_else(|| {
                format!(
                    "ordering_field `{}` is not Date32 at runtime",
                    order_plan.field_name
                )
            }),
        ParquetOrderKeyType::Date64 => column
            .as_any()
            .downcast_ref::<Date64Array>()
            .cloned()
            .map(PreparedOrderColumn::Date64)
            .ok_or_else(|| {
                format!(
                    "ordering_field `{}` is not Date64 at runtime",
                    order_plan.field_name
                )
            }),
        ParquetOrderKeyType::TimestampSecond => column
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .cloned()
            .map(PreparedOrderColumn::TimestampSecond)
            .ok_or_else(|| {
                format!(
                    "ordering_field `{}` is not Timestamp(Second) at runtime",
                    order_plan.field_name
                )
            }),
        ParquetOrderKeyType::TimestampMillisecond => column
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .cloned()
            .map(PreparedOrderColumn::TimestampMillisecond)
            .ok_or_else(|| {
                format!(
                    "ordering_field `{}` is not Timestamp(Millisecond) at runtime",
                    order_plan.field_name
                )
            }),
        ParquetOrderKeyType::TimestampMicrosecond => column
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .cloned()
            .map(PreparedOrderColumn::TimestampMicrosecond)
            .ok_or_else(|| {
                format!(
                    "ordering_field `{}` is not Timestamp(Microsecond) at runtime",
                    order_plan.field_name
                )
            }),
        ParquetOrderKeyType::TimestampNanosecond => column
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .cloned()
            .map(PreparedOrderColumn::TimestampNanosecond)
            .ok_or_else(|| {
                format!(
                    "ordering_field `{}` is not Timestamp(Nanosecond) at runtime",
                    order_plan.field_name
                )
            }),
    }
}

impl PreparedOrderBatch {
    fn new(batch: RecordBatch, order_plan: &ParquetOrderPlan) -> Result<Self, String> {
        let order_column = extract_prepared_order_column(&batch, order_plan)?;
        Ok(Self {
            batch,
            order_column,
        })
    }
}

impl PreparedOrderColumn {
    fn key_at(&self, row: usize) -> OrderKeyValue {
        match self {
            Self::Int64(array) => {
                OrderKeyValue::Int64((!array.is_null(row)).then(|| array.value(row)))
            }
            Self::UInt64(array) => {
                OrderKeyValue::UInt64((!array.is_null(row)).then(|| array.value(row)))
            }
            Self::Float64(array) => {
                OrderKeyValue::Float64((!array.is_null(row)).then(|| array.value(row)))
            }
            Self::Utf8(array) => {
                OrderKeyValue::Utf8((!array.is_null(row)).then(|| array.value(row).to_string()))
            }
            Self::LargeUtf8(array) => OrderKeyValue::LargeUtf8(
                (!array.is_null(row)).then(|| array.value(row).to_string()),
            ),
            Self::Date32(array) => {
                OrderKeyValue::Date32((!array.is_null(row)).then(|| array.value(row)))
            }
            Self::Date64(array) => {
                OrderKeyValue::Date64((!array.is_null(row)).then(|| array.value(row)))
            }
            Self::TimestampSecond(array) => {
                OrderKeyValue::Timestamp((!array.is_null(row)).then(|| array.value(row)))
            }
            Self::TimestampMillisecond(array) => {
                OrderKeyValue::Timestamp((!array.is_null(row)).then(|| array.value(row)))
            }
            Self::TimestampMicrosecond(array) => {
                OrderKeyValue::Timestamp((!array.is_null(row)).then(|| array.value(row)))
            }
            Self::TimestampNanosecond(array) => {
                OrderKeyValue::Timestamp((!array.is_null(row)).then(|| array.value(row)))
            }
        }
    }

    fn compare_rows(&self, left: usize, right: usize) -> CmpOrdering {
        match self {
            Self::Int64(array) => compare_int64_values(array, left, right),
            Self::UInt64(array) => compare_uint64_values(array, left, right),
            Self::Float64(array) => compare_float64_values(array, left, right),
            Self::Utf8(array) => compare_string_values(array, left, right),
            Self::LargeUtf8(array) => compare_large_string_values(array, left, right),
            Self::Date32(array) => compare_date32_values(array, left, right),
            Self::Date64(array) => compare_date64_values(array, left, right),
            Self::TimestampSecond(array) => compare_timestamp_second_values(array, left, right),
            Self::TimestampMillisecond(array) => {
                compare_timestamp_millisecond_values(array, left, right)
            }
            Self::TimestampMicrosecond(array) => {
                compare_timestamp_microsecond_values(array, left, right)
            }
            Self::TimestampNanosecond(array) => {
                compare_timestamp_nanosecond_values(array, left, right)
            }
        }
    }

    fn compare_row_to_value(&self, row: usize, value: &OrderKeyValue) -> CmpOrdering {
        match (self, value) {
            (Self::Int64(array), OrderKeyValue::Int64(value)) => {
                compare_int64_to_value(array, row, *value)
            }
            (Self::UInt64(array), OrderKeyValue::UInt64(value)) => {
                compare_uint64_to_value(array, row, *value)
            }
            (Self::Float64(array), OrderKeyValue::Float64(value)) => {
                compare_float64_to_value(array, row, *value)
            }
            (Self::Utf8(array), OrderKeyValue::Utf8(value)) => {
                compare_string_to_value(array, row, value.as_deref())
            }
            (Self::LargeUtf8(array), OrderKeyValue::LargeUtf8(value)) => {
                compare_large_string_to_value(array, row, value.as_deref())
            }
            (Self::Date32(array), OrderKeyValue::Date32(value)) => {
                compare_date32_to_value(array, row, *value)
            }
            (Self::Date64(array), OrderKeyValue::Date64(value)) => {
                compare_date64_to_value(array, row, *value)
            }
            (Self::TimestampSecond(array), OrderKeyValue::Timestamp(value)) => {
                compare_timestamp_second_to_value(array, row, *value)
            }
            (Self::TimestampMillisecond(array), OrderKeyValue::Timestamp(value)) => {
                compare_timestamp_millisecond_to_value(array, row, *value)
            }
            (Self::TimestampMicrosecond(array), OrderKeyValue::Timestamp(value)) => {
                compare_timestamp_microsecond_to_value(array, row, *value)
            }
            (Self::TimestampNanosecond(array), OrderKeyValue::Timestamp(value)) => {
                compare_timestamp_nanosecond_to_value(array, row, *value)
            }
            _ => CmpOrdering::Equal,
        }
    }
}

fn compare_order_key_values(left: &OrderKeyValue, right: &OrderKeyValue) -> CmpOrdering {
    match (left, right) {
        (OrderKeyValue::Int64(left), OrderKeyValue::Int64(right)) => {
            compare_option_ord(left, right)
        }
        (OrderKeyValue::UInt64(left), OrderKeyValue::UInt64(right)) => {
            compare_option_ord(left, right)
        }
        (OrderKeyValue::Float64(left), OrderKeyValue::Float64(right)) => {
            compare_option_f64(left, right)
        }
        (OrderKeyValue::Utf8(left), OrderKeyValue::Utf8(right)) => compare_option_ord(left, right),
        (OrderKeyValue::LargeUtf8(left), OrderKeyValue::LargeUtf8(right)) => {
            compare_option_ord(left, right)
        }
        (OrderKeyValue::Date32(left), OrderKeyValue::Date32(right)) => {
            compare_option_ord(left, right)
        }
        (OrderKeyValue::Date64(left), OrderKeyValue::Date64(right)) => {
            compare_option_ord(left, right)
        }
        (OrderKeyValue::Timestamp(left), OrderKeyValue::Timestamp(right)) => {
            compare_option_ord(left, right)
        }
        _ => CmpOrdering::Equal,
    }
}

fn compare_row_against_heap_entry(
    column: &PreparedOrderColumn,
    row: usize,
    competitor: &HeapEntry,
    source_index: usize,
) -> CmpOrdering {
    let ordering = column.compare_row_to_value(row, &competitor.key);
    if ordering == CmpOrdering::Equal {
        source_index.cmp(&competitor.source_index)
    } else {
        ordering
    }
}

async fn merge_payload_parquet_files_unordered(
    builders: Vec<ParquetRecordBatchStreamBuilder<File>>,
    output_path: &Path,
    mut plan: CompiledPayloadPlan,
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<(u64, u64, u64, Duration, Duration), String> {
    let execution_start = Instant::now();
    let output_file = File::create(output_path)
        .await
        .map_err(|error| format!("failed to create output parquet file: {error}"))?;
    let output_file = BufWriter::with_capacity(1 << 20, output_file);
    let writer_properties = parquet_writer_properties(execution_options);
    let mut writer = AsyncArrowWriter::try_new(
        output_file,
        plan.output_schema.clone(),
        Some(writer_properties),
    )
    .map_err(|error| format!("failed to create parquet writer: {error}"))?;

    let mut rows = 0_u64;
    let mut input_batches = 0_u64;
    let mut output_batches = 0_u64;

    for builder in builders {
        let mut stream = builder
            .with_batch_size(execution_options.read_batch_size)
            .build()
            .map_err(|error| format!("failed to build parquet stream: {error}"))?;
        while let Some(batch_result) = stream.next().await {
            let batch =
                batch_result.map_err(|error| format!("failed reading parquet batch: {error}"))?;
            input_batches += 1;
            rows += batch.num_rows() as u64;
            let adjusted_batch = plan
                .adapt_batch(&batch)
                .map_err(|error| format!("failed adapting parquet batch: {error}"))?;
            writer
                .write(&adjusted_batch)
                .await
                .map_err(|error| format!("failed writing merged parquet batch: {error}"))?;
            output_batches += 1;
        }
    }

    writer
        .close()
        .await
        .map_err(|error| format!("failed closing parquet writer: {error}"))?;

    Ok((
        rows,
        input_batches,
        output_batches,
        execution_start.elapsed(),
        Duration::default(),
    ))
}

async fn merge_payload_parquet_files_ordered(
    input_paths: &[PathBuf],
    output_path: &Path,
    output_schema: SchemaRef,
    source_adapters: Vec<Arc<CompiledSourceAdapter>>,
    order_plan: ParquetOrderPlan,
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<(u64, u64, u64, Duration, Duration), String> {
    let execution_start = Instant::now();
    let output_file = File::create(output_path)
        .await
        .map_err(|error| format!("failed to create output parquet file: {error}"))?;
    let output_file = BufWriter::with_capacity(1 << 20, output_file);
    let writer_properties = parquet_writer_properties(execution_options);
    let mut writer =
        AsyncArrowWriter::try_new(output_file, output_schema.clone(), Some(writer_properties))
            .map_err(|error| format!("failed to create parquet writer: {error}"))?;

    let order_plan = Arc::new(order_plan);
    let mut receivers = Vec::with_capacity(input_paths.len());
    let mut worker_handles = Vec::with_capacity(input_paths.len());
    for (source_index, (input_path, adapter)) in input_paths
        .iter()
        .cloned()
        .zip(source_adapters.into_iter())
        .enumerate()
    {
        let (tx, rx) = mpsc::channel(execution_options.prefetch_batches_per_source);
        let output_schema = output_schema.clone();
        let order_plan = order_plan.clone();
        let execution_options = execution_options.clone();
        let handle = tokio::spawn(async move {
            parquet_merge_source_worker(
                source_index,
                input_path,
                output_schema,
                adapter,
                order_plan,
                execution_options,
                tx,
            )
            .await
        });
        receivers.push(rx);
        worker_handles.push(handle);
    }

    let ordered_merge_start = Instant::now();
    let mut input_batches = 0_u64;
    let mut output_batches = 0_u64;
    let mut rows = 0_u64;
    let mut sources = receivers
        .into_iter()
        .enumerate()
        .map(|(source_index, receiver)| OrderedMergeSource::new(source_index, receiver))
        .collect::<Vec<_>>();
    let mut heap = BinaryHeap::new();

    for source in &mut sources {
        if source.load_next(&mut input_batches).await? {
            heap.push(
                source
                    .current_heap_entry()
                    .expect("loaded source always has a heap entry"),
            );
        }
    }

    let mut pending_batches = Vec::new();
    let mut pending_rows = 0_usize;
    while let Some(entry) = heap.pop() {
        let next_competitor = heap.peek().cloned();
        let source = &mut sources[entry.source_index];
        let remaining_output_rows = execution_options.output_batch_rows - pending_rows;
        let run_len = source.contiguous_run_len(next_competitor.as_ref(), remaining_output_rows);
        let batch = source
            .current_batch
            .as_ref()
            .expect("heap entry requires an active current batch");
        pending_batches.push(batch.batch.slice(source.current_row, run_len));
        pending_rows += run_len;
        rows += run_len as u64;
        source.current_row += run_len;

        if let Some(current_batch) = source.current_batch.as_ref() {
            if source.current_row >= current_batch.batch.num_rows() {
                if source.load_next(&mut input_batches).await? {
                    heap.push(
                        source
                            .current_heap_entry()
                            .expect("reloaded source always has a heap entry"),
                    );
                }
            } else {
                heap.push(
                    source
                        .current_heap_entry()
                        .expect("partially consumed source always has a heap entry"),
                );
            }
        }

        if pending_rows >= execution_options.output_batch_rows {
            if let Some(batch) = materialize_pending_output(&output_schema, &mut pending_batches)? {
                writer
                    .write(&batch)
                    .await
                    .map_err(|error| format!("failed writing ordered merge batch: {error}"))?;
                output_batches += 1;
            }
            pending_rows = 0;
        }
    }

    if let Some(batch) = materialize_pending_output(&output_schema, &mut pending_batches)? {
        writer
            .write(&batch)
            .await
            .map_err(|error| format!("failed writing ordered merge tail batch: {error}"))?;
        output_batches += 1;
    }

    let ordered_merge_duration = ordered_merge_start.elapsed();

    writer
        .close()
        .await
        .map_err(|error| format!("failed closing parquet writer: {error}"))?;

    for handle in worker_handles {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => return Err(error),
            Err(error) => return Err(format!("ordered merge worker failed to join: {error}")),
        }
    }

    Ok((
        rows,
        input_batches,
        output_batches,
        execution_start.elapsed(),
        ordered_merge_duration,
    ))
}

async fn parquet_merge_source_worker(
    source_index: usize,
    input_path: PathBuf,
    output_schema: SchemaRef,
    adapter: Arc<CompiledSourceAdapter>,
    order_plan: Arc<ParquetOrderPlan>,
    execution_options: ParquetMergeExecutionOptions,
    tx: mpsc::Sender<Result<PreparedOrderBatch, String>>,
) -> Result<(), String> {
    let result: Result<(), String> = async {
        let file = File::open(&input_path)
            .await
            .map_err(|error| format!("failed to open `{}`: {error}", input_path.display()))?;
        let builder = ParquetRecordBatchStreamBuilder::new(file)
            .await
            .map_err(|error| format!("failed to inspect `{}`: {error}", input_path.display()))?;
        let mut stream = builder
            .with_batch_size(execution_options.read_batch_size)
            .build()
            .map_err(|error| {
                format!(
                    "failed to build parquet stream for `{}`: {error}",
                    input_path.display()
                )
            })?;
        let mut scratch = ExecutionScratch::default();
        let mut last_key = None;

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result.map_err(|error| {
                format!(
                    "failed reading parquet batch from `{}`: {error}",
                    input_path.display()
                )
            })?;
            let adapted = adapter
                .adapt_batch(&batch, &output_schema, &mut scratch)
                .map_err(|error| {
                    format!(
                        "failed adapting parquet batch from `{}`: {error}",
                        input_path.display()
                    )
                })?;
            let prepared = PreparedOrderBatch::new(adapted, order_plan.as_ref())?;
            validate_prepared_ordered_batch(
                &prepared,
                &mut last_key,
                &input_path,
                &order_plan.field_name,
                source_index,
            )?;
            if tx.send(Ok(prepared)).await.is_err() {
                return Ok(());
            }
        }
        Ok(())
    }
    .await;

    if let Err(error) = &result {
        let _ = tx.send(Err(error.clone())).await;
    }

    result
}

fn validate_prepared_ordered_batch(
    batch: &PreparedOrderBatch,
    last_key: &mut Option<OrderKeyValue>,
    input_path: &Path,
    field_name: &str,
    source_index: usize,
) -> Result<(), String> {
    if batch.batch.num_rows() == 0 {
        return Ok(());
    }

    if let Some(previous_key) = last_key.as_ref() {
        if batch.order_column.compare_row_to_value(0, previous_key) == CmpOrdering::Less {
            return Err(format!(
                "input parquet file `{}` is not sorted ascending by `{}` for source {}",
                input_path.display(),
                field_name,
                source_index
            ));
        }
    }

    for row in 1..batch.batch.num_rows() {
        if batch.order_column.compare_rows(row - 1, row) == CmpOrdering::Greater {
            return Err(format!(
                "input parquet file `{}` is not sorted ascending by `{}` near row {}",
                input_path.display(),
                field_name,
                row
            ));
        }
    }

    *last_key = Some(batch.order_column.key_at(batch.batch.num_rows() - 1));
    Ok(())
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct DiscoveredField {
    name: InternedFieldName,
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
    field_interner: FieldInterner,
}

impl NdjsonScanState {
    fn with_capacity(read_buffer_bytes: usize) -> Self {
        Self {
            line_buffer: Vec::with_capacity(read_buffer_bytes),
            parser_buffers: Buffers::default(),
            field_interner: FieldInterner::default(),
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
    append_shape_cache: HashMap<ShapeSignature, Arc<CompiledNdjsonRecordShapeAdapter>>,
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

#[derive(Clone, Debug)]
struct CompiledNdjsonRecordShapeAdapter {
    ordered_present: Vec<CompiledNdjsonPresentRootField>,
    envelope_missing: Vec<CompiledNdjsonMissingField>,
    payload_missing: Vec<CompiledNdjsonMissingField>,
}

#[derive(Clone, Debug)]
struct CompiledNdjsonPresentRootField {
    target: RootFieldTarget,
    adapter: CompiledNdjsonValueShapeAdapter,
}

#[derive(Clone, Copy, Debug)]
enum RootFieldTarget {
    Envelope(usize),
    Payload(usize),
}

#[derive(Clone, Debug)]
struct CompiledNdjsonStructShapeAdapter {
    present: Vec<CompiledNdjsonPresentField>,
    missing: Vec<CompiledNdjsonMissingField>,
}

#[derive(Clone, Debug)]
struct CompiledNdjsonPresentField {
    builder_index: usize,
    adapter: CompiledNdjsonValueShapeAdapter,
}

#[derive(Clone, Debug)]
struct CompiledNdjsonMissingField {
    builder_index: usize,
    op: CompiledNdjsonValueOp,
}

#[derive(Clone, Debug)]
enum CompiledNdjsonValueShapeAdapter {
    Generic(CompiledNdjsonValueOp),
    Struct(CompiledNdjsonStructShapeAdapter),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SortKeyType {
    Int64,
    UInt64,
    Float64,
    Utf8,
    LargeUtf8,
}

#[derive(Clone, Debug)]
struct NdjsonSortPlan {
    field_name: String,
    field_index: usize,
    key_type: SortKeyType,
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
    let sort_plan = build_ndjson_sort_plan(planning.schema.as_ref(), options).map_err(io_error)?;
    let planning_duration = planning_start.elapsed();

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
    let execution_start = Instant::now();
    let mut buffered_batches = Vec::new();
    let mut buffered_rows = 0_usize;

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
                let batch = batch_builder.finish_batch();
                if sort_plan.is_some() {
                    buffered_rows += batch.num_rows();
                    ensure_sort_row_limit(buffered_rows, options).map_err(io_error)?;
                    buffered_batches.push(batch);
                } else {
                    writer.write(&batch).await?;
                }
            }
        }
    }

    if !batch_builder.is_empty() {
        let batch = batch_builder.finish_batch();
        if sort_plan.is_some() {
            buffered_rows += batch.num_rows();
            ensure_sort_row_limit(buffered_rows, options).map_err(io_error)?;
            buffered_batches.push(batch);
        } else {
            writer.write(&batch).await?;
        }
    }

    let accumulation_duration = execution_start.elapsed();
    let mut sorting_duration = Duration::default();
    let mut post_sort_write_duration = Duration::default();
    if let Some(sort_plan) = sort_plan.as_ref() {
        let sorting_start = Instant::now();
        let sorted_batch = if buffered_batches.is_empty() {
            RecordBatch::new_empty(planning.schema.clone())
        } else {
            let combined =
                concat_batches(&planning.schema, buffered_batches.iter()).map_err(|error| {
                    io_error(format!(
                        "failed to concatenate buffered NDJSON batches: {error}"
                    ))
                })?;
            stable_sort_record_batch(&combined, sort_plan).map_err(io_error)?
        };
        sorting_duration = sorting_start.elapsed();

        if sorted_batch.num_rows() > 0 {
            let write_start = Instant::now();
            writer.write(&sorted_batch).await?;
            post_sort_write_duration = write_start.elapsed();
        }
    }

    writer.close().await?;

    let execution_duration = accumulation_duration + post_sort_write_duration;
    let output_bytes = std::fs::metadata(output_path)?.len();

    Ok(CompactionReport {
        rows,
        input_bytes,
        output_bytes,
        planning_duration,
        execution_duration,
        sorting_duration,
        total_duration: total_start.elapsed(),
        peak_rss_bytes: peak_rss_bytes(),
        planning_threads_used: planning.stats.threads_used,
        planning_unique_shapes: planning.stats.unique_shapes,
        planning_shape_cache_hits: planning.stats.shape_cache_hits,
        planning_shape_cache_misses: planning.stats.shape_cache_misses,
        input_batches: 0,
        output_batches: 0,
        adapter_cache_hits: 0,
        adapter_cache_misses: 0,
        ordered_merge_duration: Duration::default(),
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
    let mut shape_cache: HashMap<ShapeSignature, Arc<CachedDiscovery>> = HashMap::new();
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
            let shape_key = compute_canonical_shape_signature(&value);
            if let Some(cached) = shape_cache.get(&shape_key) {
                result.shape_cache_hits += 1;
                cached.clone()
            } else {
                result.shape_cache_misses += 1;
                let discovered = Arc::new(discover_record(
                    object,
                    &envelope_lookup,
                    &mut scan_state.field_interner,
                    options,
                )?);
                shape_cache.insert(shape_key, discovered.clone());
                discovered
            }
        } else {
            result.shape_cache_misses += 1;
            Arc::new(discover_record(
                object,
                &envelope_lookup,
                &mut scan_state.field_interner,
                options,
            )?)
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
    field_interner: &mut FieldInterner,
    options: &CompactionOptions,
) -> Result<CachedDiscovery, String> {
    let mut envelope_types = vec![None; envelope_lookup.len()];
    let mut payload_fields = Vec::new();

    for (key, value) in object.iter() {
        if let Some(index) = envelope_lookup.get(key.as_ref()) {
            envelope_types[*index] = Some(discover_borrowed_type(
                value,
                field_interner,
                &options.widening_options,
            )?);
        } else {
            payload_fields.push(DiscoveredField {
                name: field_interner.intern(key.as_ref()),
                data_type: discover_borrowed_type(
                    value,
                    field_interner,
                    &options.widening_options,
                )?,
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
    field_interner: &mut FieldInterner,
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
            let discovered = discover_borrowed_type(element, field_interner, options)?;
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
                name: field_interner.intern(key.as_ref()),
                data_type: discover_borrowed_type(child, field_interner, options)?,
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
        .map(|field| (field.name.as_ref(), field))
        .collect();
    let left_names: BTreeSet<&str> = left.iter().map(|field| field.name.as_ref()).collect();

    let mut merged = Vec::with_capacity(left.len() + right.len());
    for left_field in left {
        let child_path = if path.is_empty() {
            left_field.name.as_ref().to_string()
        } else {
            format!("{path}.{}", left_field.name)
        };

        if let Some(right_field) = right_by_name.get(left_field.name.as_ref()) {
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
        if !left_names.contains(right_field.name.as_ref()) {
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
                        field.name.as_ref(),
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

fn compute_canonical_shape_signature(value: &BorrowedValue<'_>) -> ShapeSignature {
    let mut hasher = DefaultHasher::new();
    hash_canonical_shape(value, &mut hasher);
    ShapeSignature(hasher.finish())
}

fn hash_canonical_shape(value: &BorrowedValue<'_>, hasher: &mut DefaultHasher) {
    if value.is_null() {
        0_u8.hash(hasher);
    } else if value.as_bool().is_some() {
        1_u8.hash(hasher);
    } else if value.as_i64().is_some() {
        2_u8.hash(hasher);
    } else if value.as_u64().is_some() {
        3_u8.hash(hasher);
    } else if value.as_f64().is_some() {
        4_u8.hash(hasher);
    } else if value.as_str().is_some() {
        5_u8.hash(hasher);
    } else if let Some(array) = value.as_array() {
        6_u8.hash(hasher);
        let mut entries = array
            .iter()
            .map(compute_canonical_shape_signature)
            .collect::<Vec<_>>();
        entries.sort_unstable_by_key(|signature| signature.0);
        entries.dedup();
        entries.len().hash(hasher);
        for entry in entries {
            entry.hash(hasher);
        }
    } else if let Some(object) = value.as_object() {
        7_u8.hash(hasher);
        let mut entries = object
            .iter()
            .map(|(key, value)| (key.as_ref(), compute_canonical_shape_signature(value)))
            .collect::<Vec<_>>();
        entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
        entries.len().hash(hasher);
        for (key, nested) in entries {
            key.hash(hasher);
            nested.hash(hasher);
        }
    } else {
        255_u8.hash(hasher);
    }
}

fn compute_execution_shape_signature(value: &BorrowedValue<'_>) -> ShapeSignature {
    let mut hasher = DefaultHasher::new();
    hash_execution_shape(value, &mut hasher);
    ShapeSignature(hasher.finish())
}

fn hash_execution_shape(value: &BorrowedValue<'_>, hasher: &mut DefaultHasher) {
    if value.is_null() {
        0_u8.hash(hasher);
    } else if value.as_bool().is_some() {
        1_u8.hash(hasher);
    } else if value.as_i64().is_some() {
        2_u8.hash(hasher);
    } else if value.as_u64().is_some() {
        3_u8.hash(hasher);
    } else if value.as_f64().is_some() {
        4_u8.hash(hasher);
    } else if value.as_str().is_some() {
        5_u8.hash(hasher);
    } else if let Some(array) = value.as_array() {
        6_u8.hash(hasher);
        array.len().hash(hasher);
        for entry in array.iter() {
            hash_execution_shape(entry, hasher);
        }
    } else if let Some(object) = value.as_object() {
        7_u8.hash(hasher);
        object.len().hash(hasher);
        for (key, entry) in object.iter() {
            key.as_ref().hash(hasher);
            hash_execution_shape(entry, hasher);
        }
    } else {
        255_u8.hash(hasher);
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

    fn compile_record_shape_adapter(
        &self,
        record: &BorrowedValue<'_>,
    ) -> Result<Option<CompiledNdjsonRecordShapeAdapter>, String> {
        let object = record
            .as_object()
            .ok_or_else(|| "NDJSON record must be an object".to_string())?;
        let mut ordered_present = Vec::with_capacity(object.len());
        let mut envelope_seen = vec![false; self.envelope_ops.len()];
        let mut payload_seen = vec![false; self.payload_op.fields.len()];

        for (key, value) in object.iter() {
            match self.top_level_lookup.get(key.as_ref()) {
                Some(RootDispatch::Envelope(index)) => {
                    if envelope_seen[*index] {
                        return Ok(None);
                    }
                    envelope_seen[*index] = true;
                    ordered_present.push(CompiledNdjsonPresentRootField {
                        target: RootFieldTarget::Envelope(self.envelope_ops[*index].builder_index),
                        adapter: compile_value_shape_adapter(
                            &self.envelope_ops[*index].value_op,
                            value,
                        )?,
                    });
                }
                Some(RootDispatch::Payload(index)) => {
                    if payload_seen[*index] {
                        return Ok(None);
                    }
                    payload_seen[*index] = true;
                    ordered_present.push(CompiledNdjsonPresentRootField {
                        target: RootFieldTarget::Payload(
                            self.payload_op.fields[*index].builder_index,
                        ),
                        adapter: compile_value_shape_adapter(
                            &self.payload_op.fields[*index].value_op,
                            value,
                        )?,
                    });
                }
                None => return Ok(None),
            }
        }

        let envelope_missing = self
            .envelope_ops
            .iter()
            .enumerate()
            .filter(|(index, _)| !envelope_seen[*index])
            .map(|(_, op)| CompiledNdjsonMissingField {
                builder_index: op.builder_index,
                op: op.value_op.clone(),
            })
            .collect();
        let payload_missing = self
            .payload_op
            .fields
            .iter()
            .enumerate()
            .filter(|(index, _)| !payload_seen[*index])
            .map(|(_, op)| CompiledNdjsonMissingField {
                builder_index: op.builder_index,
                op: op.value_op.clone(),
            })
            .collect();

        Ok(Some(CompiledNdjsonRecordShapeAdapter {
            ordered_present,
            envelope_missing,
            payload_missing,
        }))
    }
}

fn compile_value_shape_adapter(
    op: &CompiledNdjsonValueOp,
    value: &BorrowedValue<'_>,
) -> Result<CompiledNdjsonValueShapeAdapter, String> {
    match (op, value.as_object()) {
        (CompiledNdjsonValueOp::Struct(struct_op), Some(object)) if !value.is_null() => {
            if let Some(adapter) = compile_struct_shape_adapter(struct_op, object)? {
                Ok(CompiledNdjsonValueShapeAdapter::Struct(adapter))
            } else {
                Ok(CompiledNdjsonValueShapeAdapter::Generic(op.clone()))
            }
        }
        _ => Ok(CompiledNdjsonValueShapeAdapter::Generic(op.clone())),
    }
}

fn compile_struct_shape_adapter(
    struct_op: &CompiledNdjsonStructOp,
    object: &simd_json::borrowed::Object<'_>,
) -> Result<Option<CompiledNdjsonStructShapeAdapter>, String> {
    let mut present = Vec::with_capacity(object.len());
    let mut seen = vec![false; struct_op.fields.len()];

    for (key, value) in object.iter() {
        let Some(index) = struct_op.lookup.get(key.as_ref()) else {
            return Ok(None);
        };
        if seen[*index] {
            return Ok(None);
        }
        seen[*index] = true;
        let field_op = &struct_op.fields[*index];
        present.push(CompiledNdjsonPresentField {
            builder_index: field_op.builder_index,
            adapter: compile_value_shape_adapter(&field_op.value_op, value)?,
        });
    }

    let missing = struct_op
        .fields
        .iter()
        .enumerate()
        .filter(|(index, _)| !seen[*index])
        .map(|(_, field_op)| CompiledNdjsonMissingField {
            builder_index: field_op.builder_index,
            op: field_op.value_op.clone(),
        })
        .collect();

    Ok(Some(CompiledNdjsonStructShapeAdapter { present, missing }))
}

impl CompiledNdjsonRecordShapeAdapter {
    fn can_apply(&self, record: &BorrowedValue<'_>) -> bool {
        record
            .as_object()
            .is_some_and(|object| object.len() == self.ordered_present.len())
    }

    fn apply(
        &self,
        plan: &CompiledNdjsonPlan,
        root_builder: &mut NdjsonStructBuilder,
        record: &BorrowedValue<'_>,
        scratch: &mut NdjsonAppendScratch,
    ) -> Result<(), String> {
        let object = record
            .as_object()
            .ok_or_else(|| "NDJSON record must be an object".to_string())?;
        if object.len() != self.ordered_present.len() {
            return plan.append_record(root_builder, record, scratch);
        }

        let (envelope_builders, payload_slice) = root_builder
            .children
            .split_at_mut(plan.payload_builder_index);
        let payload_builder = payload_slice
            .first_mut()
            .ok_or_else(|| "root payload builder is missing".to_string())?
            .as_struct_mut()
            .ok_or_else(|| "root payload builder must be Struct".to_string())?;

        for ((_, value), present) in object.iter().zip(self.ordered_present.iter()) {
            match present.target {
                RootFieldTarget::Envelope(builder_index) => {
                    present.adapter.apply(
                        &mut envelope_builders[builder_index],
                        Some(value),
                        scratch,
                        1,
                    )?;
                }
                RootFieldTarget::Payload(builder_index) => {
                    present.adapter.apply(
                        &mut payload_builder.children[builder_index],
                        Some(value),
                        scratch,
                        1,
                    )?;
                }
            }
        }

        for missing in &self.envelope_missing {
            missing.op.append_to(
                &mut envelope_builders[missing.builder_index],
                None,
                scratch,
                1,
            )?;
        }
        for missing in &self.payload_missing {
            missing.op.append_to(
                &mut payload_builder.children[missing.builder_index],
                None,
                scratch,
                1,
            )?;
        }

        payload_builder.append(true);
        root_builder.append(true);
        Ok(())
    }
}

impl CompiledNdjsonValueShapeAdapter {
    fn apply(
        &self,
        builder: &mut NdjsonValueBuilder,
        value: Option<&BorrowedValue<'_>>,
        scratch: &mut NdjsonAppendScratch,
        depth: usize,
    ) -> Result<(), String> {
        match self {
            Self::Generic(op) => op.append_to(builder, value, scratch, depth),
            Self::Struct(adapter) => adapter.apply(builder, value, scratch, depth),
        }
    }
}

impl CompiledNdjsonStructShapeAdapter {
    fn apply(
        &self,
        builder: &mut NdjsonValueBuilder,
        value: Option<&BorrowedValue<'_>>,
        scratch: &mut NdjsonAppendScratch,
        depth: usize,
    ) -> Result<(), String> {
        let builder = builder.as_struct_mut().ok_or_else(|| {
            "compiled NDJSON struct shape adapter expected struct builder".to_string()
        })?;

        match value {
            Some(value) if !value.is_null() => {
                let object = value.as_object().ok_or_else(|| {
                    format!(
                        "expected object for Struct field, found {}",
                        borrowed_json_kind(value)
                    )
                })?;
                if object.len() != self.present.len() {
                    return Err(
                        "cached struct shape adapter does not match object shape".to_string()
                    );
                }

                for ((_, child_value), present) in object.iter().zip(self.present.iter()) {
                    present.adapter.apply(
                        &mut builder.children[present.builder_index],
                        Some(child_value),
                        scratch,
                        depth + 1,
                    )?;
                }
                for missing in &self.missing {
                    missing.op.append_to(
                        &mut builder.children[missing.builder_index],
                        None,
                        scratch,
                        depth + 1,
                    )?;
                }
                builder.append(true);
                Ok(())
            }
            _ => {
                for present in &self.present {
                    present.adapter.apply(
                        &mut builder.children[present.builder_index],
                        None,
                        scratch,
                        depth + 1,
                    )?;
                }
                for missing in &self.missing {
                    missing.op.append_to(
                        &mut builder.children[missing.builder_index],
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
            append_shape_cache: HashMap::new(),
        })
    }

    fn append_record(&mut self, record: &BorrowedValue<'_>) -> Result<(), String> {
        let shape_signature = compute_execution_shape_signature(record);
        if let Some(adapter) = self.append_shape_cache.get(&shape_signature) {
            if adapter.can_apply(record) {
                return adapter.apply(&self.plan, &mut self.root, record, &mut self.scratch);
            }
        }

        if let Some(adapter) = self.plan.compile_record_shape_adapter(record)? {
            let adapter = Arc::new(adapter);
            let result = adapter.apply(&self.plan, &mut self.root, record, &mut self.scratch);
            self.append_shape_cache.insert(shape_signature, adapter);
            return result;
        }

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

fn build_ndjson_sort_plan(
    schema: &Schema,
    options: &CompactionOptions,
) -> Result<Option<NdjsonSortPlan>, String> {
    let Some(sort_field) = options.sort_field.as_ref() else {
        return Ok(None);
    };

    let field_index = schema
        .index_of(sort_field)
        .map_err(|_| format!("sort field `{sort_field}` is missing from the output schema"))?;
    let field = schema.field(field_index);
    let key_type = match field.data_type() {
        DataType::Int64 => SortKeyType::Int64,
        DataType::UInt64 => SortKeyType::UInt64,
        DataType::Float64 => SortKeyType::Float64,
        DataType::Utf8 => SortKeyType::Utf8,
        DataType::LargeUtf8 => SortKeyType::LargeUtf8,
        other => {
            return Err(format!(
                "sort field `{sort_field}` has unsupported type {other:?}; supported types are Int64, UInt64, Float64, Utf8, and LargeUtf8"
            ));
        }
    };

    Ok(Some(NdjsonSortPlan {
        field_name: sort_field.clone(),
        field_index,
        key_type,
    }))
}

fn ensure_sort_row_limit(row_count: usize, options: &CompactionOptions) -> Result<(), String> {
    if row_count > options.sort_max_rows_in_memory {
        return Err(format!(
            "NDJSON sort requires buffering {row_count} rows, which exceeds sort_max_rows_in_memory={}; external spill sort is not implemented",
            options.sort_max_rows_in_memory
        ));
    }
    Ok(())
}

fn stable_sort_record_batch(
    batch: &RecordBatch,
    sort_plan: &NdjsonSortPlan,
) -> Result<RecordBatch, String> {
    let column = batch.column(sort_plan.field_index);
    let mut indices = (0..batch.num_rows() as u32).collect::<Vec<_>>();

    match sort_plan.key_type {
        SortKeyType::Int64 => {
            let array = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    format!(
                        "sort field `{}` is not Int64 at runtime",
                        sort_plan.field_name
                    )
                })?;
            indices.sort_by(|left, right| {
                compare_int64_values(array, *left as usize, *right as usize)
            });
        }
        SortKeyType::UInt64 => {
            let array = column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    format!(
                        "sort field `{}` is not UInt64 at runtime",
                        sort_plan.field_name
                    )
                })?;
            indices.sort_by(|left, right| {
                compare_uint64_values(array, *left as usize, *right as usize)
            });
        }
        SortKeyType::Float64 => {
            let array = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    format!(
                        "sort field `{}` is not Float64 at runtime",
                        sort_plan.field_name
                    )
                })?;
            indices.sort_by(|left, right| {
                compare_float64_values(array, *left as usize, *right as usize)
            });
        }
        SortKeyType::Utf8 => {
            let array = column
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    format!(
                        "sort field `{}` is not Utf8 at runtime",
                        sort_plan.field_name
                    )
                })?;
            indices.sort_by(|left, right| {
                compare_string_values(array, *left as usize, *right as usize)
            });
        }
        SortKeyType::LargeUtf8 => {
            let array = column
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| {
                    format!(
                        "sort field `{}` is not LargeUtf8 at runtime",
                        sort_plan.field_name
                    )
                })?;
            indices.sort_by(|left, right| {
                compare_large_string_values(array, *left as usize, *right as usize)
            });
        }
    }

    let indices = UInt32Array::from(indices);
    take_record_batch(batch, &indices).map_err(|error| {
        format!(
            "failed to apply stable sort permutation for `{}`: {error}",
            sort_plan.field_name
        )
    })
}

fn compare_null_flags(left_null: bool, right_null: bool) -> Option<CmpOrdering> {
    match (left_null, right_null) {
        (true, true) => Some(CmpOrdering::Equal),
        (true, false) => Some(CmpOrdering::Greater),
        (false, true) => Some(CmpOrdering::Less),
        (false, false) => None,
    }
}

fn compare_int64_values(array: &Int64Array, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(&array.value(right))
}

fn compare_uint64_values(array: &UInt64Array, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(&array.value(right))
}

fn compare_float64_values(array: &Float64Array, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).total_cmp(&array.value(right))
}

fn compare_string_values(array: &StringArray, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(array.value(right))
}

fn compare_large_string_values(array: &LargeStringArray, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(array.value(right))
}

fn compare_option_ord<T: Ord>(left: &Option<T>, right: &Option<T>) -> CmpOrdering {
    match (left, right) {
        (None, None) => CmpOrdering::Equal,
        (None, Some(_)) => CmpOrdering::Greater,
        (Some(_), None) => CmpOrdering::Less,
        (Some(left), Some(right)) => left.cmp(right),
    }
}

fn compare_option_f64(left: &Option<f64>, right: &Option<f64>) -> CmpOrdering {
    match (left, right) {
        (None, None) => CmpOrdering::Equal,
        (None, Some(_)) => CmpOrdering::Greater,
        (Some(_), None) => CmpOrdering::Less,
        (Some(left), Some(right)) => left.total_cmp(right),
    }
}

fn compare_int64_to_value(array: &Int64Array, row: usize, value: Option<i64>) -> CmpOrdering {
    compare_option_ord(&((!array.is_null(row)).then(|| array.value(row))), &value)
}

fn compare_uint64_to_value(array: &UInt64Array, row: usize, value: Option<u64>) -> CmpOrdering {
    compare_option_ord(&((!array.is_null(row)).then(|| array.value(row))), &value)
}

fn compare_float64_to_value(array: &Float64Array, row: usize, value: Option<f64>) -> CmpOrdering {
    compare_option_f64(&((!array.is_null(row)).then(|| array.value(row))), &value)
}

fn compare_string_to_value(array: &StringArray, row: usize, value: Option<&str>) -> CmpOrdering {
    let left = (!array.is_null(row)).then(|| array.value(row));
    match (left, value) {
        (None, None) => CmpOrdering::Equal,
        (None, Some(_)) => CmpOrdering::Greater,
        (Some(_), None) => CmpOrdering::Less,
        (Some(left), Some(right)) => left.cmp(right),
    }
}

fn compare_large_string_to_value(
    array: &LargeStringArray,
    row: usize,
    value: Option<&str>,
) -> CmpOrdering {
    let left = (!array.is_null(row)).then(|| array.value(row));
    match (left, value) {
        (None, None) => CmpOrdering::Equal,
        (None, Some(_)) => CmpOrdering::Greater,
        (Some(_), None) => CmpOrdering::Less,
        (Some(left), Some(right)) => left.cmp(right),
    }
}

fn compare_date32_values(array: &Date32Array, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(&array.value(right))
}

fn compare_date64_values(array: &Date64Array, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(&array.value(right))
}

fn compare_date32_to_value(array: &Date32Array, row: usize, value: Option<i32>) -> CmpOrdering {
    compare_option_ord(&((!array.is_null(row)).then(|| array.value(row))), &value)
}

fn compare_date64_to_value(array: &Date64Array, row: usize, value: Option<i64>) -> CmpOrdering {
    compare_option_ord(&((!array.is_null(row)).then(|| array.value(row))), &value)
}

fn compare_timestamp_second_values(
    array: &TimestampSecondArray,
    left: usize,
    right: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(&array.value(right))
}

fn compare_timestamp_millisecond_values(
    array: &TimestampMillisecondArray,
    left: usize,
    right: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(&array.value(right))
}

fn compare_timestamp_microsecond_values(
    array: &TimestampMicrosecondArray,
    left: usize,
    right: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(&array.value(right))
}

fn compare_timestamp_nanosecond_values(
    array: &TimestampNanosecondArray,
    left: usize,
    right: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(&array.value(right))
}

fn compare_timestamp_second_to_value(
    array: &TimestampSecondArray,
    row: usize,
    value: Option<i64>,
) -> CmpOrdering {
    compare_option_ord(&((!array.is_null(row)).then(|| array.value(row))), &value)
}

fn compare_timestamp_millisecond_to_value(
    array: &TimestampMillisecondArray,
    row: usize,
    value: Option<i64>,
) -> CmpOrdering {
    compare_option_ord(&((!array.is_null(row)).then(|| array.value(row))), &value)
}

fn compare_timestamp_microsecond_to_value(
    array: &TimestampMicrosecondArray,
    row: usize,
    value: Option<i64>,
) -> CmpOrdering {
    compare_option_ord(&((!array.is_null(row)).then(|| array.value(row))), &value)
}

fn compare_timestamp_nanosecond_to_value(
    array: &TimestampNanosecondArray,
    row: usize,
    value: Option<i64>,
) -> CmpOrdering {
    compare_option_ord(&((!array.is_null(row)).then(|| array.value(row))), &value)
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
    if options.sort_max_rows_in_memory == 0 {
        return Err("sort_max_rows_in_memory must be greater than zero".to_string());
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

    if let Some(sort_field) = options.sort_field.as_ref() {
        if sort_field.trim().is_empty() {
            return Err("sort_field must not be empty when provided".to_string());
        }
        if !options
            .envelope_fields
            .iter()
            .any(|field| field == sort_field)
        {
            return Err(format!(
                "sort_field `{sort_field}` must also be listed in envelope_fields"
            ));
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

    use arrow_array::{
        BooleanArray, Date32Array, Date64Array, Float64Array, Int32Array, Int64Array,
        LargeStringArray, StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray, UInt64Array,
    };
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

    #[test]
    fn compiled_ndjson_plan_reuses_shape_cache_for_repeated_records() -> Result<(), Box<dyn Error>>
    {
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
    async fn compact_ndjson_to_parquet_rejects_unsupported_sort_type() -> Result<(), Box<dyn Error>>
    {
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
    async fn compact_ndjson_to_parquet_rejects_sort_jobs_above_memory_cap()
    -> Result<(), Box<dyn Error>> {
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
            PreparedOrderColumn::LargeUtf8(LargeStringArray::from(vec![
                Some("a"),
                Some("b"),
                None,
            ])),
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
        };

        let (_, receiver) = mpsc::channel(1);
        let lower_source = OrderedMergeSource {
            source_index: 0,
            receiver,
            current_batch: Some(prepared.clone()),
            current_row: 0,
        };
        let competitor = HeapEntry {
            source_index: 1,
            key: OrderKeyValue::Int64(Some(2)),
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
    async fn ordered_merge_widens_payload_and_preserves_stable_ties() -> Result<(), Box<dyn Error>>
    {
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

        let payload_fields: Fields =
            vec![Arc::new(Field::new("score", DataType::Int32, true))].into();
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
}
