use std::cmp::Ordering as CmpOrdering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fs::File as StdFile;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::{
    PayloadMergeOptions, TopLevelMergeOptions, WideningOptions, is_primitive_or_string,
    merge_payload_schemas_pair, merge_top_level_schemas_many, to_parquet_error, widen_data_type,
};
use arrow::array::new_null_array;
use arrow::compute::{cast, concat_batches, interleave_record_batch, take_record_batch};
use arrow_array::builder::{
    ArrayBuilder, BooleanBuilder, Float32Builder, Float64Builder, Int8Builder, Int16Builder,
    Int32Builder, Int64Builder, LargeListBuilder, LargeStringBuilder, ListBuilder, NullBuilder,
    StringBuilder, UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
};
use arrow_array::{
    Array, ArrayRef, Date32Array, Date64Array, Float64Array, Int32Array, Int64Array,
    LargeListArray, LargeStringArray, ListArray, RecordBatch, StringArray, StructArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt32Array, UInt64Array,
};
use arrow_buffer::NullBufferBuilder;
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use bytes::Bytes;
use futures_util::StreamExt;
use futures_util::future::{BoxFuture, FutureExt};
use futures_util::stream::FuturesUnordered;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, statistics::StatisticsConverter};
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_writer::AsyncArrowWriter;
use parquet::arrow::async_writer::AsyncFileWriter;
use parquet::arrow::{ArrowSchemaConverter, ArrowWriter, add_encoded_arrow_schema_to_metadata};
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::column::writer::ColumnCloseResult;
use parquet::file::metadata::{ParquetMetaDataReader, RowGroupMetaDataPtr};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use simd_json::borrowed::Value as BorrowedValue;
use simd_json::prelude::{TypedScalarValue, ValueAsArray, ValueAsObject, ValueAsScalar};
use simd_json::{Buffers, to_borrowed_value_with_buffers};
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};

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
    pub parallelism: usize,
    pub unordered_merge_order: UnorderedMergeOrder,
    pub writer_compression: ParquetCompression,
    pub writer_dictionary_enabled: bool,
    pub stats_fast_path: bool,
    pub ordered_memory_budget_bytes: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnorderedMergeOrder {
    #[default]
    PreserveInputOrder,
    AllowInterleaved,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParquetCompression {
    #[default]
    Uncompressed,
    Snappy,
    Lz4Raw,
    Zstd {
        level: i32,
    },
}

impl Default for ParquetMergeExecutionOptions {
    fn default() -> Self {
        Self {
            ordering_field: None,
            read_batch_size: 32_768,
            output_batch_rows: 32_768,
            prefetch_batches_per_source: 1,
            output_row_group_rows: 128_000,
            parallelism: 1,
            unordered_merge_order: UnorderedMergeOrder::PreserveInputOrder,
            writer_compression: ParquetCompression::Uncompressed,
            writer_dictionary_enabled: true,
            stats_fast_path: true,
            ordered_memory_budget_bytes: None,
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
    pub stats_fast_path_duration: Duration,
    pub fast_path_row_groups: u64,
    pub fast_path_batches: u64,
    pub fallback_batches: u64,
    pub read_decode_duration: Duration,
    pub source_prepare_duration: Duration,
    pub ordered_output_assembly_duration: Duration,
    pub ordered_output_selection_duration: Duration,
    pub ordered_output_materialization_duration: Duration,
    pub ordered_output_materialization_wait_duration: Duration,
    pub ordered_pipeline_peak_buffered_bytes: u64,
    pub writer_peak_buffered_bytes: u64,
    pub ordered_selector_comparisons: u64,
    pub dense_partition_jobs: u64,
    pub dense_rows: u64,
    pub dense_selection_duration: Duration,
    pub dense_materialization_duration: Duration,
    pub dense_fallback_count: u64,
    pub writer_write_duration: Duration,
    pub writer_encode_duration: Duration,
    pub writer_sink_duration: Duration,
    pub writer_close_duration: Duration,
    pub direct_batch_writes: u64,
    pub accumulator_flushes: u64,
    pub accumulator_concat_flushes: u64,
    pub accumulator_interleave_flushes: u64,
    pub copy_candidate_row_groups: u64,
    pub copied_row_groups: u64,
    pub copied_rows: u64,
    pub copied_compressed_bytes: u64,
    pub row_group_copy_duration: Duration,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RowGroupNullKind {
    NoNulls,
    AllNulls,
    #[default]
    MixedOrUnknown,
}

#[derive(Clone, Debug, PartialEq)]
struct RowGroupOrderRange {
    row_group_index: usize,
    min: Option<OrderKeyValue>,
    max: Option<OrderKeyValue>,
    null_kind: RowGroupNullKind,
}

#[derive(Clone, Debug, Default)]
struct SourceOrderMetadata {
    row_groups: Vec<RowGroupOrderRange>,
}

#[derive(Clone, Debug, Default)]
struct OrderedRowGroupCopyPlan {
    row_groups: Vec<Option<RowGroupCopyRequest>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct ParquetMergeRunStats {
    rows: u64,
    input_batches: u64,
    output_batches: u64,
    execution_duration: Duration,
    ordered_merge_duration: Duration,
    stats_fast_path_duration: Duration,
    fast_path_row_groups: u64,
    fast_path_batches: u64,
    fallback_batches: u64,
    read_decode_duration: Duration,
    source_prepare_duration: Duration,
    ordered_output_assembly_duration: Duration,
    ordered_output_selection_duration: Duration,
    ordered_output_materialization_duration: Duration,
    ordered_output_materialization_wait_duration: Duration,
    ordered_pipeline_peak_buffered_bytes: u64,
    writer_peak_buffered_bytes: u64,
    ordered_selector_comparisons: u64,
    dense_partition_jobs: u64,
    dense_rows: u64,
    dense_selection_duration: Duration,
    dense_materialization_duration: Duration,
    dense_fallback_count: u64,
    writer_write_duration: Duration,
    writer_encode_duration: Duration,
    writer_sink_duration: Duration,
    writer_close_duration: Duration,
    direct_batch_writes: u64,
    accumulator_flushes: u64,
    accumulator_concat_flushes: u64,
    accumulator_interleave_flushes: u64,
    copy_candidate_row_groups: u64,
    copied_row_groups: u64,
    copied_rows: u64,
    copied_compressed_bytes: u64,
    row_group_copy_duration: Duration,
}

#[derive(Debug, Default)]
struct ParquetWorkerMetrics {
    read_decode_nanos: AtomicU64,
    source_prepare_nanos: AtomicU64,
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
    Int32ToFloat64,
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
            Self::Int32ToFloat64 => {
                let source = array
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| format!("expected Int32Array, got {:?}", array.data_type()))?;
                Ok(Arc::new(Float64Array::from_iter(
                    source.iter().map(|value| value.map(|value| value as f64)),
                )) as ArrayRef)
            }
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

#[cfg(test)]
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct PreparedOrderBatch {
    batch: RecordBatch,
    order_column: PreparedOrderColumn,
    row_group_index: usize,
    row_group_batch_index: usize,
    non_null_prefix_len: usize,
}

#[cfg(test)]
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

trait OrderedKey: Send + Sync + 'static {
    type Array: Array + Clone + std::fmt::Debug + Send + Sync + 'static;

    const KEY_TYPE: ParquetOrderKeyType;
    const RUNTIME_NAME: &'static str;

    fn downcast(column: &ArrayRef) -> Option<&Self::Array>;
    fn compare_values(array: &Self::Array, left: usize, right: usize) -> CmpOrdering;
    fn compare_arrays(
        left: &Self::Array,
        left_row: usize,
        right: &Self::Array,
        right_row: usize,
    ) -> CmpOrdering;
    fn compare_to_value(array: &Self::Array, row: usize, value: &OrderKeyValue) -> CmpOrdering;
    fn key_at(array: &Self::Array, row: usize) -> OrderKeyValue;
    fn null_key_value() -> OrderKeyValue;

    fn is_null(array: &Self::Array, row: usize) -> bool {
        array.is_null(row)
    }

    fn non_null_prefix_len(array: &Self::Array, row_count: usize) -> usize {
        let mut low = 0;
        let mut high = row_count;
        while low < high {
            let mid = low + ((high - low) / 2);
            if Self::is_null(array, mid) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        low
    }
}

#[derive(Clone, Copy, Debug)]
struct Int64OrderKey;
#[derive(Clone, Copy, Debug)]
struct UInt64OrderKey;
#[derive(Clone, Copy, Debug)]
struct Float64OrderKey;
#[derive(Clone, Copy, Debug)]
struct Utf8OrderKey;
#[derive(Clone, Copy, Debug)]
struct LargeUtf8OrderKey;
#[derive(Clone, Copy, Debug)]
struct Date32OrderKey;
#[derive(Clone, Copy, Debug)]
struct Date64OrderKey;
#[derive(Clone, Copy, Debug)]
struct TimestampSecondOrderKey;
#[derive(Clone, Copy, Debug)]
struct TimestampMillisecondOrderKey;
#[derive(Clone, Copy, Debug)]
struct TimestampMicrosecondOrderKey;
#[derive(Clone, Copy, Debug)]
struct TimestampNanosecondOrderKey;

macro_rules! primitive_order_key {
    (
        $key:ident,
        $array:ty,
        $key_type:expr,
        $runtime_name:expr,
        $compare_values:ident,
        $compare_arrays:ident,
        $compare_to_value:ident,
        $value_variant:ident,
        $value_type:ty
    ) => {
        impl OrderedKey for $key {
            type Array = $array;

            const KEY_TYPE: ParquetOrderKeyType = $key_type;
            const RUNTIME_NAME: &'static str = $runtime_name;

            fn downcast(column: &ArrayRef) -> Option<&Self::Array> {
                column.as_any().downcast_ref::<Self::Array>()
            }

            fn compare_values(array: &Self::Array, left: usize, right: usize) -> CmpOrdering {
                $compare_values(array, left, right)
            }

            fn compare_arrays(
                left: &Self::Array,
                left_row: usize,
                right: &Self::Array,
                right_row: usize,
            ) -> CmpOrdering {
                $compare_arrays(left, left_row, right, right_row)
            }

            fn compare_to_value(
                array: &Self::Array,
                row: usize,
                value: &OrderKeyValue,
            ) -> CmpOrdering {
                match value {
                    OrderKeyValue::$value_variant(value) => $compare_to_value(array, row, *value),
                    _ => CmpOrdering::Equal,
                }
            }

            fn key_at(array: &Self::Array, row: usize) -> OrderKeyValue {
                OrderKeyValue::$value_variant(
                    (!array.is_null(row)).then(|| array.value(row) as $value_type),
                )
            }

            fn null_key_value() -> OrderKeyValue {
                OrderKeyValue::$value_variant(None)
            }
        }
    };
}

primitive_order_key!(
    Int64OrderKey,
    Int64Array,
    ParquetOrderKeyType::Int64,
    "Int64",
    compare_int64_values,
    compare_int64_arrays,
    compare_int64_to_value,
    Int64,
    i64
);
primitive_order_key!(
    UInt64OrderKey,
    UInt64Array,
    ParquetOrderKeyType::UInt64,
    "UInt64",
    compare_uint64_values,
    compare_uint64_arrays,
    compare_uint64_to_value,
    UInt64,
    u64
);
primitive_order_key!(
    Float64OrderKey,
    Float64Array,
    ParquetOrderKeyType::Float64,
    "Float64",
    compare_float64_values,
    compare_float64_arrays,
    compare_float64_to_value,
    Float64,
    f64
);
primitive_order_key!(
    Date32OrderKey,
    Date32Array,
    ParquetOrderKeyType::Date32,
    "Date32",
    compare_date32_values,
    compare_date32_arrays,
    compare_date32_to_value,
    Date32,
    i32
);
primitive_order_key!(
    Date64OrderKey,
    Date64Array,
    ParquetOrderKeyType::Date64,
    "Date64",
    compare_date64_values,
    compare_date64_arrays,
    compare_date64_to_value,
    Date64,
    i64
);

macro_rules! timestamp_order_key {
    (
        $key:ident,
        $array:ty,
        $key_type:expr,
        $runtime_name:expr,
        $compare_values:ident,
        $compare_arrays:ident,
        $compare_to_value:ident
    ) => {
        impl OrderedKey for $key {
            type Array = $array;

            const KEY_TYPE: ParquetOrderKeyType = $key_type;
            const RUNTIME_NAME: &'static str = $runtime_name;

            fn downcast(column: &ArrayRef) -> Option<&Self::Array> {
                column.as_any().downcast_ref::<Self::Array>()
            }

            fn compare_values(array: &Self::Array, left: usize, right: usize) -> CmpOrdering {
                $compare_values(array, left, right)
            }

            fn compare_arrays(
                left: &Self::Array,
                left_row: usize,
                right: &Self::Array,
                right_row: usize,
            ) -> CmpOrdering {
                $compare_arrays(left, left_row, right, right_row)
            }

            fn compare_to_value(
                array: &Self::Array,
                row: usize,
                value: &OrderKeyValue,
            ) -> CmpOrdering {
                match value {
                    OrderKeyValue::Timestamp(value) => $compare_to_value(array, row, *value),
                    _ => CmpOrdering::Equal,
                }
            }

            fn key_at(array: &Self::Array, row: usize) -> OrderKeyValue {
                OrderKeyValue::Timestamp((!array.is_null(row)).then(|| array.value(row)))
            }

            fn null_key_value() -> OrderKeyValue {
                OrderKeyValue::Timestamp(None)
            }
        }
    };
}

timestamp_order_key!(
    TimestampSecondOrderKey,
    TimestampSecondArray,
    ParquetOrderKeyType::TimestampSecond,
    "Timestamp(Second)",
    compare_timestamp_second_values,
    compare_timestamp_second_arrays,
    compare_timestamp_second_to_value
);
timestamp_order_key!(
    TimestampMillisecondOrderKey,
    TimestampMillisecondArray,
    ParquetOrderKeyType::TimestampMillisecond,
    "Timestamp(Millisecond)",
    compare_timestamp_millisecond_values,
    compare_timestamp_millisecond_arrays,
    compare_timestamp_millisecond_to_value
);
timestamp_order_key!(
    TimestampMicrosecondOrderKey,
    TimestampMicrosecondArray,
    ParquetOrderKeyType::TimestampMicrosecond,
    "Timestamp(Microsecond)",
    compare_timestamp_microsecond_values,
    compare_timestamp_microsecond_arrays,
    compare_timestamp_microsecond_to_value
);
timestamp_order_key!(
    TimestampNanosecondOrderKey,
    TimestampNanosecondArray,
    ParquetOrderKeyType::TimestampNanosecond,
    "Timestamp(Nanosecond)",
    compare_timestamp_nanosecond_values,
    compare_timestamp_nanosecond_arrays,
    compare_timestamp_nanosecond_to_value
);

impl OrderedKey for Utf8OrderKey {
    type Array = StringArray;

    const KEY_TYPE: ParquetOrderKeyType = ParquetOrderKeyType::Utf8;
    const RUNTIME_NAME: &'static str = "Utf8";

    fn downcast(column: &ArrayRef) -> Option<&Self::Array> {
        column.as_any().downcast_ref::<Self::Array>()
    }

    fn compare_values(array: &Self::Array, left: usize, right: usize) -> CmpOrdering {
        compare_string_values(array, left, right)
    }

    fn compare_arrays(
        left: &Self::Array,
        left_row: usize,
        right: &Self::Array,
        right_row: usize,
    ) -> CmpOrdering {
        compare_string_arrays(left, left_row, right, right_row)
    }

    fn compare_to_value(array: &Self::Array, row: usize, value: &OrderKeyValue) -> CmpOrdering {
        match value {
            OrderKeyValue::Utf8(value) => compare_string_to_value(array, row, value.as_deref()),
            _ => CmpOrdering::Equal,
        }
    }

    fn key_at(array: &Self::Array, row: usize) -> OrderKeyValue {
        OrderKeyValue::Utf8((!array.is_null(row)).then(|| array.value(row).to_string()))
    }

    fn null_key_value() -> OrderKeyValue {
        OrderKeyValue::Utf8(None)
    }
}

impl OrderedKey for LargeUtf8OrderKey {
    type Array = LargeStringArray;

    const KEY_TYPE: ParquetOrderKeyType = ParquetOrderKeyType::LargeUtf8;
    const RUNTIME_NAME: &'static str = "LargeUtf8";

    fn downcast(column: &ArrayRef) -> Option<&Self::Array> {
        column.as_any().downcast_ref::<Self::Array>()
    }

    fn compare_values(array: &Self::Array, left: usize, right: usize) -> CmpOrdering {
        compare_large_string_values(array, left, right)
    }

    fn compare_arrays(
        left: &Self::Array,
        left_row: usize,
        right: &Self::Array,
        right_row: usize,
    ) -> CmpOrdering {
        compare_large_string_arrays(left, left_row, right, right_row)
    }

    fn compare_to_value(array: &Self::Array, row: usize, value: &OrderKeyValue) -> CmpOrdering {
        match value {
            OrderKeyValue::LargeUtf8(value) => {
                compare_large_string_to_value(array, row, value.as_deref())
            }
            _ => CmpOrdering::Equal,
        }
    }

    fn key_at(array: &Self::Array, row: usize) -> OrderKeyValue {
        OrderKeyValue::LargeUtf8((!array.is_null(row)).then(|| array.value(row).to_string()))
    }

    fn null_key_value() -> OrderKeyValue {
        OrderKeyValue::LargeUtf8(None)
    }
}

#[derive(Clone, Debug)]
struct PreparedOrderBatchTyped<K: OrderedKey> {
    batch: RecordBatch,
    order_column: K::Array,
    row_group_index: usize,
    row_group_batch_index: usize,
    non_null_prefix_len: usize,
    _key: PhantomData<K>,
}

impl<K: OrderedKey> PreparedOrderBatchTyped<K> {
    fn new(
        batch: RecordBatch,
        order_plan: &ParquetOrderPlan,
        row_group_index: usize,
        row_group_batch_index: usize,
    ) -> Result<Self, String> {
        let column = batch.column(order_plan.field_index);
        let order_column = K::downcast(column).cloned().ok_or_else(|| {
            format!(
                "ordering_field `{}` is not {} at runtime",
                order_plan.field_name,
                K::RUNTIME_NAME
            )
        })?;
        let non_null_prefix_len = K::non_null_prefix_len(&order_column, batch.num_rows());
        Ok(Self {
            batch,
            order_column,
            row_group_index,
            row_group_batch_index,
            non_null_prefix_len,
            _key: PhantomData,
        })
    }
}

#[cfg(test)]
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
struct SourceHeadHeap {
    indices: Vec<usize>,
}

#[cfg(test)]
#[allow(dead_code)]
#[derive(Debug)]
struct OrderedMergeSource {
    source_index: usize,
    receiver: mpsc::Receiver<Result<PreparedOrderBatch, String>>,
    current_batch: Option<PreparedOrderBatch>,
    current_row: usize,
}

#[cfg(test)]
#[allow(dead_code)]
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

    fn current_row_group_index(&self) -> Option<usize> {
        self.current_batch
            .as_ref()
            .map(|batch| batch.row_group_index)
    }

    fn has_non_null_remaining(&self) -> bool {
        let Some(batch) = self.current_batch.as_ref() else {
            return false;
        };
        self.current_row < batch.non_null_prefix_len
    }

    fn compare_head_to_source(&self, other: &Self) -> CmpOrdering {
        let self_batch = self
            .current_batch
            .as_ref()
            .expect("source head comparison requires an active batch");
        let other_batch = other
            .current_batch
            .as_ref()
            .expect("source head comparison requires an active batch");
        let ordering = self_batch.order_column.compare_row_to_other(
            self.current_row,
            &other_batch.order_column,
            other.current_row,
        );
        if ordering == CmpOrdering::Equal {
            self.source_index.cmp(&other.source_index)
        } else {
            ordering
        }
    }

    fn compare_row_to_source(&self, row: usize, other: &Self) -> CmpOrdering {
        let self_batch = self
            .current_batch
            .as_ref()
            .expect("row comparison requires an active batch");
        let other_batch = other
            .current_batch
            .as_ref()
            .expect("row comparison requires an active batch");
        let ordering = self_batch.order_column.compare_row_to_other(
            row,
            &other_batch.order_column,
            other.current_row,
        );
        if ordering == CmpOrdering::Equal {
            self.source_index.cmp(&other.source_index)
        } else {
            ordering
        }
    }

    fn contiguous_run_len(&self, next_competitor: Option<&Self>, max_rows: usize) -> usize {
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

        if limit == 1 {
            return 1;
        }

        if self.compare_row_to_source(self.current_row + 1, competitor) == CmpOrdering::Greater {
            return 1;
        }

        let mut low = 2;
        let mut high = limit;
        while low < high {
            let mid = low + ((high - low + 1) / 2);
            let row = self.current_row + mid - 1;
            if self.compare_row_to_source(row, competitor) == CmpOrdering::Greater {
                high = mid - 1;
            } else {
                low = mid;
            }
        }
        low
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeapEntry {
    source_index: usize,
}

#[derive(Clone, Debug)]
struct OrderedRowGroupCandidate {
    row_group_index: usize,
    range: RowGroupOrderRange,
    copy: RowGroupCopyRequest,
}

#[derive(Debug)]
enum OrderedSourceMessage<K: OrderedKey> {
    Batch(PreparedOrderBatchTyped<K>),
    CopyCandidate(OrderedRowGroupCandidate),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrderedSourceCommand {
    Decode,
    Skip,
}

#[derive(Debug)]
struct OrderedMergeSourceTyped<K: OrderedKey> {
    source_index: usize,
    receiver: mpsc::Receiver<Result<OrderedSourceMessage<K>, String>>,
    command_tx: mpsc::Sender<OrderedSourceCommand>,
    current_batch: Option<PreparedOrderBatchTyped<K>>,
    current_candidate: Option<OrderedRowGroupCandidate>,
    current_row: usize,
}

impl<K: OrderedKey> OrderedMergeSourceTyped<K> {
    fn new(
        source_index: usize,
        receiver: mpsc::Receiver<Result<OrderedSourceMessage<K>, String>>,
        command_tx: mpsc::Sender<OrderedSourceCommand>,
    ) -> Self {
        Self {
            source_index,
            receiver,
            command_tx,
            current_batch: None,
            current_candidate: None,
            current_row: 0,
        }
    }

    async fn load_next(&mut self, input_batches: &mut u64) -> Result<bool, String> {
        match self.receiver.recv().await {
            Some(Ok(OrderedSourceMessage::Batch(batch))) => {
                *input_batches += 1;
                self.current_batch = Some(batch);
                self.current_candidate = None;
                self.current_row = 0;
                Ok(true)
            }
            Some(Ok(OrderedSourceMessage::CopyCandidate(candidate))) => {
                self.current_batch = None;
                self.current_candidate = Some(candidate);
                self.current_row = 0;
                Ok(true)
            }
            Some(Err(error)) => Err(error),
            None => {
                self.current_batch = None;
                self.current_candidate = None;
                self.current_row = 0;
                Ok(false)
            }
        }
    }

    async fn decode_current_candidate(&mut self, input_batches: &mut u64) -> Result<bool, String> {
        if self.current_candidate.is_none() {
            return Ok(self.current_batch.is_some());
        }
        self.command_tx
            .send(OrderedSourceCommand::Decode)
            .await
            .map_err(|_| "ordered source worker closed before decoding row group".to_string())?;
        self.load_next(input_batches).await
    }

    async fn skip_current_candidate(&mut self, input_batches: &mut u64) -> Result<bool, String> {
        if self.current_candidate.is_none() {
            return Ok(self.current_batch.is_some());
        }
        self.command_tx
            .send(OrderedSourceCommand::Skip)
            .await
            .map_err(|_| "ordered source worker closed before skipping row group".to_string())?;
        self.load_next(input_batches).await
    }

    fn current_row_group_index(&self) -> Option<usize> {
        self.current_batch
            .as_ref()
            .map(|batch| batch.row_group_index)
            .or_else(|| {
                self.current_candidate
                    .as_ref()
                    .map(|candidate| candidate.row_group_index)
            })
    }

    fn has_non_null_remaining(&self) -> bool {
        if let Some(candidate) = self.current_candidate.as_ref() {
            return candidate.range.null_kind == RowGroupNullKind::NoNulls;
        }
        let Some(batch) = self.current_batch.as_ref() else {
            return false;
        };
        self.current_row < batch.non_null_prefix_len
    }

    fn may_have_non_null_remaining(&self) -> bool {
        if let Some(candidate) = self.current_candidate.as_ref() {
            return !matches!(candidate.range.null_kind, RowGroupNullKind::AllNulls);
        }
        self.has_non_null_remaining()
    }

    fn has_active_head(&self) -> bool {
        self.current_batch.is_some() || self.current_candidate.is_some()
    }

    fn compare_head_to_source(&self, other: &Self) -> CmpOrdering {
        self.compare_row_to_source(self.current_row, other)
    }

    fn compare_row_to_source(&self, row: usize, other: &Self) -> CmpOrdering {
        let self_batch = self
            .current_batch
            .as_ref()
            .expect("row comparison requires an active batch");
        let other_batch = other
            .current_batch
            .as_ref()
            .expect("row comparison requires an active batch");
        let ordering = K::compare_arrays(
            &self_batch.order_column,
            row,
            &other_batch.order_column,
            other.current_row,
        );
        if ordering == CmpOrdering::Equal {
            self.source_index.cmp(&other.source_index)
        } else {
            ordering
        }
    }

    fn contiguous_run_len(&self, next_competitor: Option<&Self>, max_rows: usize) -> usize {
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

        if limit == 1 {
            return 1;
        }

        if self.compare_row_to_source(self.current_row + 1, competitor) == CmpOrdering::Greater {
            return 1;
        }

        let mut low = 2;
        let mut high = limit;
        while low < high {
            let mid = low + ((high - low + 1) / 2);
            let row = self.current_row + mid - 1;
            if self.compare_row_to_source(row, competitor) == CmpOrdering::Greater {
                high = mid - 1;
            } else {
                low = mid;
            }
        }
        low
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TypedSourceTournament<K: OrderedKey> {
    tree_size: usize,
    tree: Vec<Option<usize>>,
    active: Vec<bool>,
    active_count: usize,
    comparisons: u64,
    _key: PhantomData<K>,
}

impl<K: OrderedKey> TypedSourceTournament<K> {
    fn new(source_count: usize) -> Self {
        let tree_size = source_count.max(1).next_power_of_two();
        Self {
            tree_size,
            tree: vec![None; tree_size * 2],
            active: vec![false; source_count],
            active_count: 0,
            comparisons: 0,
            _key: PhantomData,
        }
    }

    fn is_empty(&self) -> bool {
        self.active_count == 0
    }

    fn winner(&self) -> Option<HeapEntry> {
        self.tree
            .get(1)
            .and_then(|source| *source)
            .map(|source_index| HeapEntry { source_index })
    }

    fn set_active(
        &mut self,
        source_index: usize,
        active: bool,
        sources: &[OrderedMergeSourceTyped<K>],
    ) {
        if self.active[source_index] == active {
            if active {
                self.recompute_path(source_index, sources);
            }
            return;
        }
        self.active[source_index] = active;
        if active {
            self.active_count += 1;
        } else {
            self.active_count -= 1;
        }
        let leaf = self.tree_size + source_index;
        self.tree[leaf] = active.then_some(source_index);
        self.recompute_path(source_index, sources);
    }

    fn remove(&mut self, source_index: usize, sources: &[OrderedMergeSourceTyped<K>]) -> bool {
        if !self.active.get(source_index).copied().unwrap_or(false) {
            return false;
        }
        self.set_active(source_index, false, sources);
        true
    }

    fn next_competitor(
        &mut self,
        winner_index: usize,
        sources: &[OrderedMergeSourceTyped<K>],
    ) -> Option<HeapEntry> {
        if !self.active.get(winner_index).copied().unwrap_or(false) {
            return None;
        }
        let mut position = self.tree_size + winner_index;
        let mut best = None;
        while position > 1 {
            let sibling = if position % 2 == 0 {
                position + 1
            } else {
                position - 1
            };
            if let Some(candidate) = self.tree.get(sibling).and_then(|source| *source) {
                best = Some(match best {
                    None => candidate,
                    Some(current) => self.pick_winner(candidate, current, sources),
                });
            }
            position /= 2;
        }
        best.map(|source_index| HeapEntry { source_index })
    }

    fn take_comparisons(&mut self) -> u64 {
        std::mem::take(&mut self.comparisons)
    }

    fn recompute_path(&mut self, source_index: usize, sources: &[OrderedMergeSourceTyped<K>]) {
        let mut position = (self.tree_size + source_index) / 2;
        while position > 0 {
            let left = self.tree[position * 2];
            let right = self.tree[(position * 2) + 1];
            self.tree[position] = match (left, right) {
                (Some(left), Some(right)) => Some(self.pick_winner(left, right, sources)),
                (Some(left), None) => Some(left),
                (None, Some(right)) => Some(right),
                (None, None) => None,
            };
            if position == 1 {
                break;
            }
            position /= 2;
        }
    }

    fn pick_winner(
        &mut self,
        left_index: usize,
        right_index: usize,
        sources: &[OrderedMergeSourceTyped<K>],
    ) -> usize {
        self.comparisons += 1;
        if sources[left_index].compare_head_to_source(&sources[right_index]) == CmpOrdering::Less {
            left_index
        } else {
            right_index
        }
    }
}

#[cfg(test)]
#[allow(dead_code)]
impl SourceHeadHeap {
    fn new() -> Self {
        Self {
            indices: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    fn push(&mut self, source_index: usize, sources: &[OrderedMergeSource]) {
        self.indices.push(source_index);
        let last = self.indices.len() - 1;
        self.sift_up(last, sources);
    }

    fn pop(&mut self, sources: &[OrderedMergeSource]) -> Option<HeapEntry> {
        if self.indices.is_empty() {
            return None;
        }
        let source_index = self.indices.swap_remove(0);
        if !self.indices.is_empty() {
            self.sift_down(0, sources);
        }
        Some(HeapEntry { source_index })
    }

    fn peek(&self) -> Option<HeapEntry> {
        self.indices
            .first()
            .copied()
            .map(|source_index| HeapEntry { source_index })
    }

    fn remove(&mut self, source_index: usize, sources: &[OrderedMergeSource]) -> bool {
        let Some(position) = self
            .indices
            .iter()
            .position(|candidate| *candidate == source_index)
        else {
            return false;
        };
        self.indices.swap_remove(position);
        if position < self.indices.len() {
            self.sift_down(position, sources);
            self.sift_up(position, sources);
        }
        true
    }

    fn sift_up(&mut self, mut index: usize, sources: &[OrderedMergeSource]) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.less(index, parent, sources) {
                self.indices.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut index: usize, sources: &[OrderedMergeSource]) {
        loop {
            let left = (index * 2) + 1;
            if left >= self.indices.len() {
                break;
            }
            let right = left + 1;
            let mut smallest = left;
            if right < self.indices.len() && self.less(right, left, sources) {
                smallest = right;
            }
            if self.less(smallest, index, sources) {
                self.indices.swap(index, smallest);
                index = smallest;
            } else {
                break;
            }
        }
    }

    fn less(&self, left_pos: usize, right_pos: usize, sources: &[OrderedMergeSource]) -> bool {
        let left = self.indices[left_pos];
        let right = self.indices[right_pos];
        compare_source_heads(sources, left, right) == CmpOrdering::Less
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

fn build_compiled_top_level_plan<'a, I>(
    schemas: I,
    options: &TopLevelMergeOptions,
) -> Result<CompiledPayloadPlan, String>
where
    I: IntoIterator<Item = &'a Schema>,
{
    let output_schema = Arc::new(merge_top_level_schemas_many(schemas, options)?);
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

pub async fn merge_top_level_parquet_files(
    input_paths: &[PathBuf],
    output_path: &Path,
    options: &TopLevelMergeOptions,
) -> Result<CompactionReport, Box<dyn Error>> {
    merge_top_level_parquet_files_with_execution(
        input_paths,
        output_path,
        options,
        &ParquetMergeExecutionOptions::default(),
    )
    .await
}

pub async fn merge_top_level_parquet_files_with_execution(
    input_paths: &[PathBuf],
    output_path: &Path,
    options: &TopLevelMergeOptions,
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<CompactionReport, Box<dyn Error>> {
    if input_paths.is_empty() {
        return Err(io_error("at least one parquet input path is required").into());
    }
    validate_parquet_merge_execution_options(execution_options).map_err(io_error)?;
    if execution_options.ordering_field.is_some() {
        return Err(io_error(
            "top-level parquet merge does not support ordering_field; use unordered execution options",
        )
        .into());
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

    let mut plan =
        build_compiled_top_level_plan(schemas.iter().map(|schema| schema.as_ref()), options)?;

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
    let run_stats = merge_parquet_files_unordered(
        builders,
        output_path,
        plan.output_schema.clone(),
        source_adapters,
        execution_options,
    )
    .await
    .map_err(io_error)?;
    let output_bytes = std::fs::metadata(output_path)?.len();

    Ok(CompactionReport {
        rows: run_stats.rows,
        input_bytes,
        output_bytes,
        planning_duration,
        execution_duration: run_stats.execution_duration,
        sorting_duration: Duration::default(),
        total_duration: total_start.elapsed(),
        peak_rss_bytes: peak_rss_bytes(),
        planning_threads_used: 0,
        planning_unique_shapes: 0,
        planning_shape_cache_hits: 0,
        planning_shape_cache_misses: 0,
        input_batches: run_stats.input_batches,
        output_batches: run_stats.output_batches,
        adapter_cache_hits,
        adapter_cache_misses,
        ordered_merge_duration: run_stats.ordered_merge_duration,
        stats_fast_path_duration: run_stats.stats_fast_path_duration,
        fast_path_row_groups: run_stats.fast_path_row_groups,
        fast_path_batches: run_stats.fast_path_batches,
        fallback_batches: run_stats.fallback_batches,
        read_decode_duration: run_stats.read_decode_duration,
        source_prepare_duration: run_stats.source_prepare_duration,
        ordered_output_assembly_duration: run_stats.ordered_output_assembly_duration,
        ordered_output_selection_duration: run_stats.ordered_output_selection_duration,
        ordered_output_materialization_duration: run_stats.ordered_output_materialization_duration,
        ordered_output_materialization_wait_duration: run_stats
            .ordered_output_materialization_wait_duration,
        ordered_pipeline_peak_buffered_bytes: run_stats.ordered_pipeline_peak_buffered_bytes,
        writer_peak_buffered_bytes: run_stats.writer_peak_buffered_bytes,
        ordered_selector_comparisons: run_stats.ordered_selector_comparisons,
        dense_partition_jobs: run_stats.dense_partition_jobs,
        dense_rows: run_stats.dense_rows,
        dense_selection_duration: run_stats.dense_selection_duration,
        dense_materialization_duration: run_stats.dense_materialization_duration,
        dense_fallback_count: run_stats.dense_fallback_count,
        writer_write_duration: run_stats.writer_write_duration,
        writer_encode_duration: run_stats.writer_encode_duration,
        writer_sink_duration: run_stats.writer_sink_duration,
        writer_close_duration: run_stats.writer_close_duration,
        direct_batch_writes: run_stats.direct_batch_writes,
        accumulator_flushes: run_stats.accumulator_flushes,
        accumulator_concat_flushes: run_stats.accumulator_concat_flushes,
        accumulator_interleave_flushes: run_stats.accumulator_interleave_flushes,
        copy_candidate_row_groups: run_stats.copy_candidate_row_groups,
        copied_row_groups: run_stats.copied_row_groups,
        copied_rows: run_stats.copied_rows,
        copied_compressed_bytes: run_stats.copied_compressed_bytes,
        row_group_copy_duration: run_stats.row_group_copy_duration,
    })
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
    let source_order_metadata = if execution_options.stats_fast_path {
        if let Some(order_plan) = order_plan.as_ref() {
            let mut metadata = Vec::with_capacity(builders.len());
            for builder in &builders {
                metadata.push(build_source_order_metadata(builder, order_plan)?);
            }
            metadata
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let planning_duration = planning_start.elapsed();
    let run_stats = if let Some(order_plan) = order_plan.as_ref() {
        merge_payload_parquet_files_ordered(
            input_paths,
            output_path,
            plan.output_schema.clone(),
            schemas.clone(),
            source_adapters,
            source_order_metadata,
            order_plan.clone(),
            execution_options,
        )
        .await
        .map_err(io_error)?
    } else {
        merge_parquet_files_unordered(
            builders,
            output_path,
            plan.output_schema.clone(),
            source_adapters,
            execution_options,
        )
        .await
        .map_err(io_error)?
    };

    let output_bytes = std::fs::metadata(output_path)?.len();

    Ok(CompactionReport {
        rows: run_stats.rows,
        input_bytes,
        output_bytes,
        planning_duration,
        execution_duration: run_stats.execution_duration,
        sorting_duration: Duration::default(),
        total_duration: total_start.elapsed(),
        peak_rss_bytes: peak_rss_bytes(),
        planning_threads_used: 0,
        planning_unique_shapes: 0,
        planning_shape_cache_hits: 0,
        planning_shape_cache_misses: 0,
        input_batches: run_stats.input_batches,
        output_batches: run_stats.output_batches,
        adapter_cache_hits,
        adapter_cache_misses,
        ordered_merge_duration: run_stats.ordered_merge_duration,
        stats_fast_path_duration: run_stats.stats_fast_path_duration,
        fast_path_row_groups: run_stats.fast_path_row_groups,
        fast_path_batches: run_stats.fast_path_batches,
        fallback_batches: run_stats.fallback_batches,
        read_decode_duration: run_stats.read_decode_duration,
        source_prepare_duration: run_stats.source_prepare_duration,
        ordered_output_assembly_duration: run_stats.ordered_output_assembly_duration,
        ordered_output_selection_duration: run_stats.ordered_output_selection_duration,
        ordered_output_materialization_duration: run_stats.ordered_output_materialization_duration,
        ordered_output_materialization_wait_duration: run_stats
            .ordered_output_materialization_wait_duration,
        ordered_pipeline_peak_buffered_bytes: run_stats.ordered_pipeline_peak_buffered_bytes,
        writer_peak_buffered_bytes: run_stats.writer_peak_buffered_bytes,
        ordered_selector_comparisons: run_stats.ordered_selector_comparisons,
        dense_partition_jobs: run_stats.dense_partition_jobs,
        dense_rows: run_stats.dense_rows,
        dense_selection_duration: run_stats.dense_selection_duration,
        dense_materialization_duration: run_stats.dense_materialization_duration,
        dense_fallback_count: run_stats.dense_fallback_count,
        writer_write_duration: run_stats.writer_write_duration,
        writer_encode_duration: run_stats.writer_encode_duration,
        writer_sink_duration: run_stats.writer_sink_duration,
        writer_close_duration: run_stats.writer_close_duration,
        direct_batch_writes: run_stats.direct_batch_writes,
        accumulator_flushes: run_stats.accumulator_flushes,
        accumulator_concat_flushes: run_stats.accumulator_concat_flushes,
        accumulator_interleave_flushes: run_stats.accumulator_interleave_flushes,
        copy_candidate_row_groups: run_stats.copy_candidate_row_groups,
        copied_row_groups: run_stats.copied_row_groups,
        copied_rows: run_stats.copied_rows,
        copied_compressed_bytes: run_stats.copied_compressed_bytes,
        row_group_copy_duration: run_stats.row_group_copy_duration,
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
    if options.ordered_memory_budget_bytes == Some(0) {
        return Err("ordered_memory_budget_bytes must be greater than zero when set".to_string());
    }
    if let ParquetCompression::Zstd { level } = options.writer_compression {
        if !(1..=22).contains(&level) {
            return Err(format!(
                "writer_compression zstd level must be between 1 and 22, got {level}"
            ));
        }
    }
    if let Some(ordering_field) = options.ordering_field.as_ref() {
        if ordering_field.trim().is_empty() {
            return Err("ordering_field must not be empty when provided".to_string());
        }
    }
    Ok(())
}

fn resolve_parquet_parallelism(
    options: &ParquetMergeExecutionOptions,
    input_count: usize,
) -> usize {
    if input_count == 0 {
        return 0;
    }

    let requested = if options.parallelism == 0 {
        default_scan_parallelism()
    } else {
        options.parallelism
    };
    requested.max(1).min(input_count)
}

fn resolve_parquet_cpu_parallelism(options: &ParquetMergeExecutionOptions) -> usize {
    let requested = if options.parallelism == 0 {
        default_scan_parallelism()
    } else {
        options.parallelism
    };
    requested.max(1)
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

fn build_source_order_metadata(
    builder: &ParquetRecordBatchStreamBuilder<File>,
    order_plan: &ParquetOrderPlan,
) -> Result<SourceOrderMetadata, String> {
    let metadata = builder.metadata();
    let row_groups = metadata.row_groups();
    let converter = StatisticsConverter::try_new(
        &order_plan.field_name,
        builder.schema().as_ref(),
        metadata.file_metadata().schema_descr(),
    )
    .map_err(|error| {
        format!(
            "failed building statistics converter for `{}`: {error}",
            order_plan.field_name
        )
    })?;
    let min_values = converter
        .row_group_mins(row_groups.iter())
        .map_err(|error| format!("failed reading row group minimum statistics: {error}"))?;
    let max_values = converter
        .row_group_maxes(row_groups.iter())
        .map_err(|error| format!("failed reading row group maximum statistics: {error}"))?;
    let min_exact = converter
        .row_group_is_min_value_exact(row_groups.iter())
        .map_err(|error| format!("failed reading row group minimum exactness: {error}"))?;
    let max_exact = converter
        .row_group_is_max_value_exact(row_groups.iter())
        .map_err(|error| format!("failed reading row group maximum exactness: {error}"))?;
    let null_counts = converter
        .with_missing_null_counts_as_zero(false)
        .row_group_null_counts(row_groups.iter())
        .map_err(|error| format!("failed reading row group null counts: {error}"))?;

    let mut ranges = Vec::with_capacity(row_groups.len());
    for (row_group_index, row_group) in row_groups.iter().enumerate() {
        let row_count = row_group.num_rows() as u64;
        let null_kind = if null_counts.is_null(row_group_index) {
            RowGroupNullKind::MixedOrUnknown
        } else {
            let null_count = null_counts.value(row_group_index);
            if null_count == 0 {
                RowGroupNullKind::NoNulls
            } else if null_count >= row_count {
                RowGroupNullKind::AllNulls
            } else {
                RowGroupNullKind::MixedOrUnknown
            }
        };

        let min = if !min_exact.is_null(row_group_index) && min_exact.value(row_group_index) {
            extract_stats_order_key(&min_values, row_group_index, order_plan.key_type)?
        } else {
            None
        };
        let max = if !max_exact.is_null(row_group_index) && max_exact.value(row_group_index) {
            extract_stats_order_key(&max_values, row_group_index, order_plan.key_type)?
        } else {
            None
        };

        ranges.push(RowGroupOrderRange {
            row_group_index,
            min,
            max,
            null_kind,
        });
    }

    Ok(SourceOrderMetadata { row_groups: ranges })
}

fn build_ordered_row_group_copy_plans(
    input_paths: &[PathBuf],
    source_schemas: &[SchemaRef],
    output_schema: &Schema,
    source_adapters: &[Arc<CompiledSourceAdapter>],
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<Vec<OrderedRowGroupCopyPlan>, String> {
    let (output_parquet_schema, _) =
        parquet_file_writer_schema_and_properties(output_schema, execution_options)?;
    let requested_compression = parquet_compression(execution_options.writer_compression)
        .map_err(|error| format!("invalid parquet writer compression: {error}"))?;
    input_paths
        .iter()
        .zip(source_schemas.iter())
        .zip(source_adapters.iter())
        .map(|((input_path, source_schema), adapter)| {
            let file = StdFile::open(input_path)
                .map_err(|error| format!("failed opening parquet metadata for copy: {error}"))?;
            let metadata = ParquetMetaDataReader::new()
                .parse_and_finish(&file)
                .map_err(|error| format!("failed reading parquet metadata for copy: {error}"))?;
            let source_path = Arc::new(input_path.clone());
            let file_copy_safe = adapter.identical_schema
                && source_schema.as_ref() == output_schema
                && metadata.file_metadata().schema() == output_parquet_schema.as_ref();
            let row_groups = metadata
                .row_groups()
                .iter()
                .map(|row_group| {
                    let row_count = row_group.num_rows();
                    if !file_copy_safe
                        || row_count < 0
                        || !row_group_copy_matches_writer_properties(
                            row_group,
                            requested_compression,
                            execution_options.writer_dictionary_enabled,
                        )
                    {
                        return None;
                    }
                    let compressed_bytes = row_group
                        .columns()
                        .iter()
                        .map(|column| column.compressed_size().max(0) as u64)
                        .sum();
                    Some(RowGroupCopyRequest {
                        source_path: source_path.clone(),
                        row_group_metadata: row_group.clone().into(),
                        rows: row_count as u64,
                        compressed_bytes,
                    })
                })
                .collect();
            Ok(OrderedRowGroupCopyPlan { row_groups })
        })
        .collect()
}

fn row_group_copy_matches_writer_properties(
    row_group: &parquet::file::metadata::RowGroupMetaData,
    requested_compression: Compression,
    writer_dictionary_enabled: bool,
) -> bool {
    row_group.columns().iter().all(|column| {
        column.compression() == requested_compression
            && (writer_dictionary_enabled
                || !column.encodings().iter().any(|encoding| {
                    matches!(
                        encoding,
                        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                    )
                }))
    })
}

fn extract_stats_order_key(
    values: &ArrayRef,
    index: usize,
    key_type: ParquetOrderKeyType,
) -> Result<Option<OrderKeyValue>, String> {
    match key_type {
        ParquetOrderKeyType::Int64 => values
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|array| {
                (!array.is_null(index)).then(|| OrderKeyValue::Int64(Some(array.value(index))))
            })
            .ok_or_else(|| "ordering statistics array is not Int64".to_string()),
        ParquetOrderKeyType::UInt64 => values
            .as_any()
            .downcast_ref::<UInt64Array>()
            .map(|array| {
                (!array.is_null(index)).then(|| OrderKeyValue::UInt64(Some(array.value(index))))
            })
            .ok_or_else(|| "ordering statistics array is not UInt64".to_string()),
        ParquetOrderKeyType::Float64 => values
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|array| {
                (!array.is_null(index)).then(|| OrderKeyValue::Float64(Some(array.value(index))))
            })
            .ok_or_else(|| "ordering statistics array is not Float64".to_string()),
        ParquetOrderKeyType::Utf8 => values
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|array| {
                (!array.is_null(index))
                    .then(|| OrderKeyValue::Utf8(Some(array.value(index).to_string())))
            })
            .ok_or_else(|| "ordering statistics array is not Utf8".to_string()),
        ParquetOrderKeyType::LargeUtf8 => values
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|array| {
                (!array.is_null(index))
                    .then(|| OrderKeyValue::LargeUtf8(Some(array.value(index).to_string())))
            })
            .ok_or_else(|| "ordering statistics array is not LargeUtf8".to_string()),
        ParquetOrderKeyType::Date32 => values
            .as_any()
            .downcast_ref::<Date32Array>()
            .map(|array| {
                (!array.is_null(index)).then(|| OrderKeyValue::Date32(Some(array.value(index))))
            })
            .ok_or_else(|| "ordering statistics array is not Date32".to_string()),
        ParquetOrderKeyType::Date64 => values
            .as_any()
            .downcast_ref::<Date64Array>()
            .map(|array| {
                (!array.is_null(index)).then(|| OrderKeyValue::Date64(Some(array.value(index))))
            })
            .ok_or_else(|| "ordering statistics array is not Date64".to_string()),
        ParquetOrderKeyType::TimestampSecond => values
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .map(|array| {
                (!array.is_null(index)).then(|| OrderKeyValue::Timestamp(Some(array.value(index))))
            })
            .ok_or_else(|| "ordering statistics array is not Timestamp(Second)".to_string()),
        ParquetOrderKeyType::TimestampMillisecond => values
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .map(|array| {
                (!array.is_null(index)).then(|| OrderKeyValue::Timestamp(Some(array.value(index))))
            })
            .ok_or_else(|| "ordering statistics array is not Timestamp(Millisecond)".to_string()),
        ParquetOrderKeyType::TimestampMicrosecond => values
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .map(|array| {
                (!array.is_null(index)).then(|| OrderKeyValue::Timestamp(Some(array.value(index))))
            })
            .ok_or_else(|| "ordering statistics array is not Timestamp(Microsecond)".to_string()),
        ParquetOrderKeyType::TimestampNanosecond => values
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .map(|array| {
                (!array.is_null(index)).then(|| OrderKeyValue::Timestamp(Some(array.value(index))))
            })
            .ok_or_else(|| "ordering statistics array is not Timestamp(Nanosecond)".to_string()),
    }
}

fn parquet_compression(
    compression: ParquetCompression,
) -> Result<Compression, parquet::errors::ParquetError> {
    match compression {
        ParquetCompression::Uncompressed => Ok(Compression::UNCOMPRESSED),
        ParquetCompression::Snappy => Ok(Compression::SNAPPY),
        ParquetCompression::Lz4Raw => Ok(Compression::LZ4_RAW),
        ParquetCompression::Zstd { level } => ZstdLevel::try_new(level).map(Compression::ZSTD),
    }
}

fn parquet_writer_properties(
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<WriterProperties, String> {
    let compression = parquet_compression(execution_options.writer_compression)
        .map_err(|error| format!("invalid parquet writer compression: {error}"))?;
    Ok(WriterProperties::builder()
        .set_max_row_group_size(execution_options.output_row_group_rows)
        .set_write_batch_size(execution_options.output_batch_rows)
        .set_compression(compression)
        .set_dictionary_enabled(execution_options.writer_dictionary_enabled)
        .build())
}

fn parquet_file_writer_schema_and_properties(
    output_schema: &Schema,
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<(parquet::schema::types::TypePtr, WriterProperties), String> {
    let parquet_schema = ArrowSchemaConverter::new()
        .convert(output_schema)
        .map_err(|error| format!("failed converting arrow schema to parquet schema: {error}"))?;
    let mut writer_properties = parquet_writer_properties(execution_options)?;
    add_encoded_arrow_schema_to_metadata(output_schema, &mut writer_properties);
    Ok((parquet_schema.root_schema_ptr(), writer_properties))
}

#[derive(Clone, Debug)]
struct OrderedBatchIdentity {
    rows: usize,
    columns: Vec<usize>,
}

#[derive(Clone, Debug)]
struct OrderedBatchSource {
    key: OrderedBatchIdentity,
    batch: RecordBatch,
}

#[derive(Clone, Debug)]
struct OrderedBatchFragment {
    source_index: usize,
    start: usize,
    len: usize,
}

#[derive(Debug)]
struct OrderedOutputAccumulator {
    schema: SchemaRef,
    batch_sources: Vec<OrderedBatchSource>,
    fragments: Vec<OrderedBatchFragment>,
    rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrderedFlushMode {
    Direct,
    Concat,
    Interleave,
}

#[derive(Debug)]
struct OrderedFlush {
    batch: RecordBatch,
    mode: OrderedFlushMode,
}

#[derive(Debug)]
enum OrderedFlushSegment {
    Completed(OrderedFlush),
    Materialize(OrderedMaterializationJob),
}

#[derive(Debug)]
struct OrderedMaterializationJob {
    schema: SchemaRef,
    batch_sources: Vec<OrderedBatchSource>,
    fragments: Vec<OrderedBatchFragment>,
    rows: usize,
    mode: OrderedFlushMode,
}

impl OrderedMaterializationJob {
    fn estimated_bytes(&self) -> usize {
        self.batch_sources
            .iter()
            .map(|source| source.batch.get_array_memory_size())
            .sum::<usize>()
            .max(self.rows.saturating_mul(8))
    }

    fn materialize(self) -> Result<OrderedFlush, String> {
        match self.mode {
            OrderedFlushMode::Direct => Err(
                "direct ordered flushes should not be scheduled for materialization".to_string(),
            ),
            OrderedFlushMode::Interleave => {
                let batch_refs = self
                    .batch_sources
                    .iter()
                    .map(|source| &source.batch)
                    .collect::<Vec<_>>();
                let mut indices = Vec::with_capacity(self.rows);
                for fragment in self.fragments {
                    for row in fragment.start..fragment.start + fragment.len {
                        indices.push((fragment.source_index, row));
                    }
                }
                interleave_record_batch(&batch_refs, &indices)
                    .map(|batch| OrderedFlush {
                        batch,
                        mode: OrderedFlushMode::Interleave,
                    })
                    .map_err(|error| format!("failed to interleave ordered output batch: {error}"))
            }
            OrderedFlushMode::Concat => {
                let batches = self
                    .fragments
                    .into_iter()
                    .map(|fragment| {
                        self.batch_sources[fragment.source_index]
                            .batch
                            .slice(fragment.start, fragment.len)
                    })
                    .collect::<Vec<_>>();

                concat_batches(&self.schema, batches.iter())
                    .map(|batch| OrderedFlush {
                        batch,
                        mode: OrderedFlushMode::Concat,
                    })
                    .map_err(|error| format!("failed to materialize ordered output batch: {error}"))
            }
        }
    }
}

impl OrderedFlush {
    fn estimated_bytes(&self) -> usize {
        self.batch.get_array_memory_size()
    }
}

impl OrderedFlushSegment {
    fn estimated_bytes(&self) -> usize {
        match self {
            Self::Completed(flush) => flush.estimated_bytes(),
            Self::Materialize(job) => job.estimated_bytes(),
        }
    }
}

impl OrderedOutputAccumulator {
    const CONCAT_FRAGMENT_LIMIT: usize = 8;
    const CONCAT_MIN_AVG_FRAGMENT_ROWS: usize = 1024;

    fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            batch_sources: Vec::new(),
            fragments: Vec::new(),
            rows: 0,
        }
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn is_empty(&self) -> bool {
        self.rows == 0
    }

    fn append_range(
        &mut self,
        batch: &RecordBatch,
        start: usize,
        len: usize,
    ) -> Result<(), String> {
        if len == 0 {
            return Ok(());
        }

        if start + len > batch.num_rows() {
            return Err(format!(
                "ordered accumulator append range out of bounds: start={start}, len={len}, rows={}",
                batch.num_rows()
            ));
        }
        if batch.schema().fields().len() != self.schema.fields().len() {
            return Err(format!(
                "ordered accumulator schema mismatch: expected {} columns, got {}",
                self.schema.fields().len(),
                batch.schema().fields().len()
            ));
        }

        let batch_key = ordered_batch_identity(batch);
        let source_index = self
            .batch_sources
            .iter()
            .position(|source| source.key == batch_key)
            .unwrap_or_else(|| {
                let next = self.batch_sources.len();
                self.batch_sources.push(OrderedBatchSource {
                    key: batch_key,
                    batch: batch.clone(),
                });
                next
            });
        if let Some(previous) = self.fragments.last_mut() {
            if previous.source_index == source_index && previous.start + previous.len == start {
                previous.len += len;
                self.rows += len;
                return Ok(());
            }
        }
        self.fragments.push(OrderedBatchFragment {
            source_index,
            start,
            len,
        });
        self.rows += len;
        Ok(())
    }

    #[cfg(test)]
    fn flush(&mut self) -> Result<Option<RecordBatch>, String> {
        Ok(self.flush_with_mode()?.map(|flushed| flushed.batch))
    }

    #[cfg(test)]
    fn flush_with_mode(&mut self) -> Result<Option<OrderedFlush>, String> {
        let Some(segment) = self.flush_segment()? else {
            return Ok(None);
        };
        match segment {
            OrderedFlushSegment::Completed(flush) => Ok(Some(flush)),
            OrderedFlushSegment::Materialize(job) => job.materialize().map(Some),
        }
    }

    fn flush_segment(&mut self) -> Result<Option<OrderedFlushSegment>, String> {
        if self.rows == 0 {
            return Ok(None);
        }

        let rows = std::mem::take(&mut self.rows);
        if self.fragments.len() == 1 {
            let fragment = self
                .fragments
                .pop()
                .expect("single fragment exists for direct materialization");
            let batch = self
                .batch_sources
                .pop()
                .expect("single fragment has a backing source batch")
                .batch;
            let batch = if fragment.start == 0 && fragment.len == batch.num_rows() {
                batch
            } else {
                batch.slice(fragment.start, fragment.len)
            };
            return Ok(Some(OrderedFlushSegment::Completed(OrderedFlush {
                batch,
                mode: OrderedFlushMode::Direct,
            })));
        }

        let batch_sources = std::mem::take(&mut self.batch_sources);
        let fragments = std::mem::take(&mut self.fragments);
        let mode = if Self::should_interleave(&fragments, rows) {
            OrderedFlushMode::Interleave
        } else {
            OrderedFlushMode::Concat
        };
        Ok(Some(OrderedFlushSegment::Materialize(
            OrderedMaterializationJob {
                schema: self.schema.clone(),
                batch_sources,
                fragments,
                rows,
                mode,
            },
        )))
    }

    fn should_interleave(fragments: &[OrderedBatchFragment], rows: usize) -> bool {
        if fragments.len() <= Self::CONCAT_FRAGMENT_LIMIT {
            return false;
        }
        rows / fragments.len().max(1) < Self::CONCAT_MIN_AVG_FRAGMENT_ROWS
    }
}

impl PartialEq for OrderedBatchIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.columns == other.columns
    }
}

impl Eq for OrderedBatchIdentity {}

fn ordered_batch_identity(batch: &RecordBatch) -> OrderedBatchIdentity {
    OrderedBatchIdentity {
        rows: batch.num_rows(),
        columns: batch
            .columns()
            .iter()
            .map(|column| Arc::as_ptr(column) as *const () as usize)
            .collect(),
    }
}

const ORDERED_MEMORY_PERMIT_BYTES: usize = 1 << 20;

#[derive(Debug)]
struct OrderedMemoryLimiter {
    semaphore: Arc<Semaphore>,
    bytes_per_permit: usize,
    total_permits: usize,
    current_permits: AtomicUsize,
    peak_permits: AtomicUsize,
}

impl OrderedMemoryLimiter {
    fn new(budget_bytes: usize) -> Arc<Self> {
        let total_permits = budget_bytes
            .div_ceil(ORDERED_MEMORY_PERMIT_BYTES)
            .max(1)
            .min(u32::MAX as usize);
        Arc::new(Self {
            semaphore: Arc::new(Semaphore::new(total_permits)),
            bytes_per_permit: ORDERED_MEMORY_PERMIT_BYTES,
            total_permits,
            current_permits: AtomicUsize::new(0),
            peak_permits: AtomicUsize::new(0),
        })
    }

    #[cfg(test)]
    async fn acquire(self: &Arc<Self>, bytes: usize) -> Result<OrderedMemoryPermit, String> {
        let permits = self.permits_for(bytes);
        let permit = self
            .semaphore
            .clone()
            .acquire_many_owned(permits as u32)
            .await
            .map_err(|_| "ordered memory budget semaphore closed".to_string())?;
        let current = self.current_permits.fetch_add(permits, Ordering::Relaxed) + permits;
        self.update_peak(current);
        Ok(OrderedMemoryPermit {
            _permit: Some(permit),
            limiter: Some(self.clone()),
            permits,
        })
    }

    fn try_acquire(self: &Arc<Self>, bytes: usize) -> Option<OrderedMemoryPermit> {
        let permits = self.permits_for(bytes);
        let permit = self
            .semaphore
            .clone()
            .try_acquire_many_owned(permits as u32)
            .ok()?;
        let current = self.current_permits.fetch_add(permits, Ordering::Relaxed) + permits;
        self.update_peak(current);
        Some(OrderedMemoryPermit {
            _permit: Some(permit),
            limiter: Some(self.clone()),
            permits,
        })
    }

    fn permits_for(&self, bytes: usize) -> usize {
        bytes
            .max(1)
            .div_ceil(self.bytes_per_permit)
            .max(1)
            .min(self.total_permits)
    }

    fn update_peak(&self, current: usize) {
        let mut observed = self.peak_permits.load(Ordering::Relaxed);
        while current > observed {
            match self.peak_permits.compare_exchange_weak(
                observed,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => observed = actual,
            }
        }
    }

    fn peak_bytes(&self) -> u64 {
        self.peak_permits.load(Ordering::Relaxed) as u64 * self.bytes_per_permit as u64
    }
}

#[derive(Debug)]
struct OrderedMemoryPermit {
    _permit: Option<OwnedSemaphorePermit>,
    limiter: Option<Arc<OrderedMemoryLimiter>>,
    permits: usize,
}

impl OrderedMemoryPermit {
    fn none() -> Self {
        Self {
            _permit: None,
            limiter: None,
            permits: 0,
        }
    }
}

impl Drop for OrderedMemoryPermit {
    fn drop(&mut self) {
        if let Some(limiter) = &self.limiter {
            limiter
                .current_permits
                .fetch_sub(self.permits, Ordering::Relaxed);
        }
    }
}

#[derive(Debug)]
enum OrderedCompletedOutputItem {
    Batch {
        flush: OrderedFlush,
        count_materialization: bool,
    },
    CopyRowGroup(RowGroupCopyRequest),
}

#[derive(Debug)]
struct OrderedCompletedOutput {
    sequence: u64,
    item: OrderedCompletedOutputItem,
    materialization_duration: Duration,
    memory_permit: OrderedMemoryPermit,
    estimated_bytes: usize,
}

type OrderedMaterializationHandle = tokio::task::JoinHandle<Result<OrderedCompletedOutput, String>>;

#[derive(Debug)]
struct OrderedOutputPipeline {
    pending: FuturesUnordered<OrderedMaterializationHandle>,
    completed: BTreeMap<u64, OrderedCompletedOutput>,
    next_sequence: u64,
    next_to_send: u64,
    limit: usize,
    memory_limiter: Option<Arc<OrderedMemoryLimiter>>,
}

impl OrderedOutputPipeline {
    fn new(limit: usize, memory_limiter: Option<Arc<OrderedMemoryLimiter>>) -> Self {
        Self {
            pending: FuturesUnordered::new(),
            completed: BTreeMap::new(),
            next_sequence: 0,
            next_to_send: 0,
            limit: limit.max(1),
            memory_limiter,
        }
    }

    async fn submit_flush_segment(
        &mut self,
        segment: OrderedFlushSegment,
        writer: &mut ParquetOutputWriter,
        stats: &mut ParquetMergeRunStats,
        direct_materialization_duration: Duration,
    ) -> Result<(), String> {
        self.ensure_capacity(writer, stats).await?;
        let sequence = self.next_sequence();
        let estimated_bytes = segment.estimated_bytes();
        let memory_permit = self.acquire_memory(estimated_bytes, writer, stats).await?;
        match segment {
            OrderedFlushSegment::Completed(flush) => {
                self.record_completed(
                    OrderedCompletedOutput {
                        sequence,
                        item: OrderedCompletedOutputItem::Batch {
                            flush,
                            count_materialization: true,
                        },
                        materialization_duration: direct_materialization_duration,
                        memory_permit,
                        estimated_bytes,
                    },
                    writer,
                    stats,
                )
                .await
            }
            OrderedFlushSegment::Materialize(job) => {
                self.pending.push(tokio::task::spawn_blocking(move || {
                    let materialization_start = Instant::now();
                    let flush = job.materialize()?;
                    let estimated_bytes = flush.estimated_bytes();
                    Ok(OrderedCompletedOutput {
                        sequence,
                        item: OrderedCompletedOutputItem::Batch {
                            flush,
                            count_materialization: true,
                        },
                        materialization_duration: materialization_start.elapsed(),
                        memory_permit,
                        estimated_bytes,
                    })
                }));
                self.flush_ready(writer, stats).await
            }
        }
    }

    async fn submit_direct_batch(
        &mut self,
        batch: RecordBatch,
        writer: &mut ParquetOutputWriter,
        stats: &mut ParquetMergeRunStats,
    ) -> Result<(), String> {
        self.ensure_capacity(writer, stats).await?;
        let sequence = self.next_sequence();
        let estimated_bytes = batch.get_array_memory_size();
        let memory_permit = self.acquire_memory(estimated_bytes, writer, stats).await?;
        self.record_completed(
            OrderedCompletedOutput {
                sequence,
                item: OrderedCompletedOutputItem::Batch {
                    flush: OrderedFlush {
                        batch,
                        mode: OrderedFlushMode::Direct,
                    },
                    count_materialization: false,
                },
                materialization_duration: Duration::default(),
                memory_permit,
                estimated_bytes,
            },
            writer,
            stats,
        )
        .await
    }

    async fn submit_materialized_batch(
        &mut self,
        batch: RecordBatch,
        mode: OrderedFlushMode,
        materialization_duration: Duration,
        writer: &mut ParquetOutputWriter,
        stats: &mut ParquetMergeRunStats,
    ) -> Result<(), String> {
        self.ensure_capacity(writer, stats).await?;
        let sequence = self.next_sequence();
        let estimated_bytes = batch.get_array_memory_size();
        let memory_permit = self.acquire_memory(estimated_bytes, writer, stats).await?;
        self.record_completed(
            OrderedCompletedOutput {
                sequence,
                item: OrderedCompletedOutputItem::Batch {
                    flush: OrderedFlush { batch, mode },
                    count_materialization: true,
                },
                materialization_duration,
                memory_permit,
                estimated_bytes,
            },
            writer,
            stats,
        )
        .await
    }

    async fn submit_row_group_copy(
        &mut self,
        copy: RowGroupCopyRequest,
        writer: &mut ParquetOutputWriter,
        stats: &mut ParquetMergeRunStats,
    ) -> Result<(), String> {
        self.ensure_capacity(writer, stats).await?;
        let sequence = self.next_sequence();
        let estimated_bytes = copy.compressed_bytes as usize;
        let memory_permit = self.acquire_memory(estimated_bytes, writer, stats).await?;
        self.record_completed(
            OrderedCompletedOutput {
                sequence,
                item: OrderedCompletedOutputItem::CopyRowGroup(copy),
                materialization_duration: Duration::default(),
                memory_permit,
                estimated_bytes,
            },
            writer,
            stats,
        )
        .await
    }

    async fn finish(
        &mut self,
        writer: &mut ParquetOutputWriter,
        stats: &mut ParquetMergeRunStats,
    ) -> Result<(), String> {
        while !self.pending.is_empty() {
            self.drain_one(writer, stats).await?;
        }
        self.flush_ready(writer, stats).await?;
        if !self.completed.is_empty() {
            return Err("ordered output pipeline finished with unsent output".to_string());
        }
        Ok(())
    }

    fn next_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        sequence
    }

    async fn acquire_memory(
        &mut self,
        estimated_bytes: usize,
        writer: &mut ParquetOutputWriter,
        stats: &mut ParquetMergeRunStats,
    ) -> Result<OrderedMemoryPermit, String> {
        match self.memory_limiter.clone() {
            Some(limiter) => loop {
                if let Some(permit) = limiter.try_acquire(estimated_bytes) {
                    return Ok(permit);
                }
                if self.pending.is_empty() {
                    // The ordered budget is a soft queue budget. If downstream writer
                    // stages already hold the permits, waiting here can deadlock the
                    // producer before the writer receives enough rows to flush.
                    return Ok(OrderedMemoryPermit::none());
                }
                self.drain_one(writer, stats).await?;
            },
            None => Ok(OrderedMemoryPermit::none()),
        }
    }

    async fn ensure_capacity(
        &mut self,
        writer: &mut ParquetOutputWriter,
        stats: &mut ParquetMergeRunStats,
    ) -> Result<(), String> {
        self.flush_ready(writer, stats).await?;
        while self.pending.len() + self.completed.len() >= self.limit && !self.pending.is_empty() {
            self.drain_one(writer, stats).await?;
        }
        self.flush_ready(writer, stats).await
    }

    async fn drain_one(
        &mut self,
        writer: &mut ParquetOutputWriter,
        stats: &mut ParquetMergeRunStats,
    ) -> Result<(), String> {
        let wait_start = Instant::now();
        let Some(join_result) = self.pending.next().await else {
            return Ok(());
        };
        stats.ordered_output_materialization_wait_duration += wait_start.elapsed();
        let completed = match join_result {
            Ok(Ok(completed)) => completed,
            Ok(Err(error)) => return Err(error),
            Err(error) => {
                return Err(format!(
                    "ordered materialization task failed to join: {error}"
                ));
            }
        };
        self.record_completed(completed, writer, stats).await
    }

    async fn record_completed(
        &mut self,
        completed: OrderedCompletedOutput,
        writer: &mut ParquetOutputWriter,
        stats: &mut ParquetMergeRunStats,
    ) -> Result<(), String> {
        if let OrderedCompletedOutputItem::Batch {
            flush,
            count_materialization: true,
        } = &completed.item
        {
            add_ordered_materialization_duration(
                stats,
                completed.materialization_duration,
                flush.mode,
            );
        }
        self.completed.insert(completed.sequence, completed);
        self.flush_ready(writer, stats).await
    }

    async fn flush_ready(
        &mut self,
        writer: &mut ParquetOutputWriter,
        stats: &mut ParquetMergeRunStats,
    ) -> Result<(), String> {
        while let Some(completed) = self.completed.remove(&self.next_to_send) {
            let OrderedCompletedOutput {
                item,
                memory_permit,
                estimated_bytes,
                ..
            } = completed;
            match item {
                OrderedCompletedOutputItem::Batch { flush, .. } => {
                    writer
                        .send_with_memory(flush.batch, memory_permit, estimated_bytes)
                        .await?;
                }
                OrderedCompletedOutputItem::CopyRowGroup(copy) => {
                    writer
                        .send_row_group_copy_with_memory(copy, memory_permit, estimated_bytes)
                        .await?;
                }
            }
            self.next_to_send += 1;
        }
        stats.ordered_output_assembly_duration = stats
            .ordered_output_selection_duration
            .saturating_add(stats.ordered_output_materialization_duration);
        Ok(())
    }
}

#[derive(Debug, Default)]
struct ParquetWriterResult {
    output_batches: u64,
    write_duration: Duration,
    encode_duration: Duration,
    sink_duration: Duration,
    close_duration: Duration,
    copied_row_groups: u64,
    copied_rows: u64,
    copied_compressed_bytes: u64,
    row_group_copy_duration: Duration,
    writer_peak_buffered_bytes: u64,
}

#[derive(Debug, Default)]
struct ParquetAsyncSinkMetrics {
    sink_nanos: AtomicU64,
}

impl ParquetAsyncSinkMetrics {
    fn add_sink_duration(&self, duration: Duration) {
        self.sink_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    fn sink_duration(&self) -> Duration {
        Duration::from_nanos(self.sink_nanos.load(Ordering::Relaxed))
    }
}

#[derive(Debug)]
struct TimedAsyncFileWriter {
    inner: BufWriter<File>,
    metrics: Arc<ParquetAsyncSinkMetrics>,
}

impl TimedAsyncFileWriter {
    fn new(inner: BufWriter<File>, metrics: Arc<ParquetAsyncSinkMetrics>) -> Self {
        Self { inner, metrics }
    }
}

impl AsyncFileWriter for TimedAsyncFileWriter {
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, parquet::errors::Result<()>> {
        async move {
            let start = Instant::now();
            self.inner.write_all(&bs).await?;
            self.metrics.add_sink_duration(start.elapsed());
            Ok(())
        }
        .boxed()
    }

    fn complete(&mut self) -> BoxFuture<'_, parquet::errors::Result<()>> {
        async move {
            let start = Instant::now();
            self.inner.flush().await?;
            self.inner.shutdown().await?;
            self.metrics.add_sink_duration(start.elapsed());
            Ok(())
        }
        .boxed()
    }
}

type TimedAsyncArrowWriter = AsyncArrowWriter<TimedAsyncFileWriter>;

#[derive(Debug)]
struct RowGroupEncodeJob {
    sequence: u64,
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    rows: usize,
    estimated_bytes: usize,
    _memory_permits: Vec<OrderedMemoryPermit>,
    writer_properties: WriterProperties,
}

#[derive(Clone, Debug)]
struct RowGroupCopyRequest {
    source_path: Arc<PathBuf>,
    row_group_metadata: RowGroupMetaDataPtr,
    rows: u64,
    compressed_bytes: u64,
}

#[derive(Debug)]
enum EncodedRowGroupData {
    InMemory(Bytes),
    Copied(RowGroupCopyRequest),
}

#[derive(Debug)]
struct EncodedRowGroup {
    sequence: u64,
    data: EncodedRowGroupData,
    row_group_metadata: parquet::file::metadata::RowGroupMetaDataPtr,
    encode_duration: Duration,
    estimated_bytes: usize,
    _memory_permits: Vec<OrderedMemoryPermit>,
}

#[derive(Debug)]
struct RowGroupBatchAccumulator {
    schema: SchemaRef,
    target_rows: usize,
    writer_properties: WriterProperties,
    batches: Vec<RecordBatch>,
    rows: usize,
    buffered_bytes: usize,
    memory_permits: Vec<OrderedMemoryPermit>,
    next_sequence: u64,
}

impl RowGroupBatchAccumulator {
    fn new(schema: SchemaRef, target_rows: usize, writer_properties: WriterProperties) -> Self {
        Self {
            schema,
            target_rows,
            writer_properties,
            batches: Vec::new(),
            rows: 0,
            buffered_bytes: 0,
            memory_permits: Vec::new(),
            next_sequence: 0,
        }
    }

    fn push(
        &mut self,
        batch: RecordBatch,
        memory_permit: OrderedMemoryPermit,
    ) -> Vec<RowGroupEncodeJob> {
        let mut jobs = Vec::new();
        let mut offset = 0;
        self.memory_permits.push(memory_permit);
        while offset < batch.num_rows() {
            let available = self.target_rows - self.rows;
            let take_rows = available.min(batch.num_rows() - offset);
            let slice = if offset == 0 && take_rows == batch.num_rows() {
                batch.clone()
            } else {
                batch.slice(offset, take_rows)
            };
            self.buffered_bytes = self
                .buffered_bytes
                .saturating_add(slice.get_array_memory_size());
            self.batches.push(slice);
            self.rows += take_rows;
            offset += take_rows;

            if self.rows == self.target_rows {
                jobs.push(self.flush());
            }
        }
        jobs
    }

    fn finish(&mut self) -> Option<RowGroupEncodeJob> {
        (self.rows > 0).then(|| self.flush())
    }

    fn next_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        sequence
    }

    fn flush(&mut self) -> RowGroupEncodeJob {
        let sequence = self.next_sequence();
        RowGroupEncodeJob {
            sequence,
            schema: self.schema.clone(),
            batches: std::mem::take(&mut self.batches),
            rows: std::mem::take(&mut self.rows),
            estimated_bytes: std::mem::take(&mut self.buffered_bytes),
            _memory_permits: std::mem::take(&mut self.memory_permits),
            writer_properties: self.writer_properties.clone(),
        }
    }
}

type RowGroupEncodeHandle = tokio::task::JoinHandle<Result<EncodedRowGroup, String>>;

fn spawn_row_group_encoder(
    pending: &mut FuturesUnordered<RowGroupEncodeHandle>,
    job: RowGroupEncodeJob,
) {
    pending.push(tokio::task::spawn_blocking(move || encode_row_group(job)));
}

fn encode_row_group(job: RowGroupEncodeJob) -> Result<EncodedRowGroup, String> {
    let encode_start = Instant::now();
    let mut writer = ArrowWriter::try_new(Vec::new(), job.schema, Some(job.writer_properties))
        .map_err(|error| format!("failed creating row group encoder: {error}"))?;
    for batch in &job.batches {
        writer
            .write(batch)
            .map_err(|error| format!("failed encoding row group batch: {error}"))?;
    }
    let bytes = Bytes::from(
        writer
            .into_inner()
            .map_err(|error| format!("failed taking encoded row group bytes: {error}"))?,
    );
    let metadata = ParquetMetaDataReader::new()
        .parse_and_finish(&bytes)
        .map_err(|error| format!("failed reading encoded row group metadata: {error}"))?;
    if metadata.row_groups().len() != 1 {
        return Err(format!(
            "expected exactly one encoded row group, got {}",
            metadata.row_groups().len()
        ));
    }
    let row_group_metadata = metadata.row_groups()[0].clone();
    if row_group_metadata.num_rows() != job.rows as i64 {
        return Err(format!(
            "encoded row group row count mismatch: expected {}, got {}",
            job.rows,
            row_group_metadata.num_rows()
        ));
    }
    Ok(EncodedRowGroup {
        sequence: job.sequence,
        data: EncodedRowGroupData::InMemory(bytes),
        row_group_metadata: row_group_metadata.into(),
        encode_duration: encode_start.elapsed(),
        estimated_bytes: job.estimated_bytes,
        _memory_permits: job._memory_permits,
    })
}

fn append_encoded_row_group<W: Write + Send>(
    writer: &mut SerializedFileWriter<W>,
    encoded: EncodedRowGroup,
) -> Result<bool, String> {
    let mut row_group_writer = writer
        .next_row_group()
        .map_err(|error| format!("failed creating output row group: {error}"))?;
    let rows_written = u64::try_from(encoded.row_group_metadata.num_rows())
        .map_err(|_| "encoded row group row count does not fit u64".to_string())?;
    let copied = matches!(encoded.data, EncodedRowGroupData::Copied(_));
    let source_file = match &encoded.data {
        EncodedRowGroupData::InMemory(_) => None,
        EncodedRowGroupData::Copied(copy) => Some(
            StdFile::open(copy.source_path.as_ref())
                .map_err(|error| format!("failed opening copied row group source: {error}"))?,
        ),
    };
    for column in encoded.row_group_metadata.columns() {
        let result = ColumnCloseResult {
            bytes_written: column.compressed_size() as u64,
            rows_written,
            metadata: column.clone(),
            bloom_filter: None,
            column_index: None,
            offset_index: None,
        };
        match (&encoded.data, source_file.as_ref()) {
            (EncodedRowGroupData::InMemory(bytes), _) => row_group_writer
                .append_column(bytes, result)
                .map_err(|error| format!("failed appending encoded column chunk: {error}"))?,
            (EncodedRowGroupData::Copied(_), Some(file)) => row_group_writer
                .append_column(file, result)
                .map_err(|error| format!("failed appending copied column chunk: {error}"))?,
            (EncodedRowGroupData::Copied(_), None) => {
                return Err("copied row group source file was not opened".to_string());
            }
        }
    }
    row_group_writer
        .close()
        .map_err(|error| format!("failed closing output row group: {error}"))?;
    Ok(copied)
}

#[derive(Debug)]
enum ParquetOutputItem {
    Batch {
        batch: RecordBatch,
        memory_permit: OrderedMemoryPermit,
        estimated_bytes: usize,
    },
    CopyRowGroup {
        copy: RowGroupCopyRequest,
        memory_permit: OrderedMemoryPermit,
        estimated_bytes: usize,
    },
}

#[derive(Debug)]
struct ParquetOutputWriter {
    tx: Option<mpsc::Sender<ParquetOutputItem>>,
    handle: Option<tokio::task::JoinHandle<Result<ParquetWriterResult, String>>>,
    supports_row_group_copy: bool,
}

impl ParquetOutputWriter {
    fn new_serial(
        writer: TimedAsyncArrowWriter,
        sink_metrics: Arc<ParquetAsyncSinkMetrics>,
        channel_capacity: usize,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<ParquetOutputItem>(channel_capacity.max(1));
        let handle = tokio::spawn(async move {
            let mut writer = writer;
            let mut result = ParquetWriterResult::default();
            while let Some(item) = rx.recv().await {
                let ParquetOutputItem::Batch {
                    batch,
                    memory_permit: _memory_permit,
                    estimated_bytes,
                } = item
                else {
                    return Err(
                        "encoded row group copy requires the parallel parquet writer path"
                            .to_string(),
                    );
                };
                result.writer_peak_buffered_bytes = result
                    .writer_peak_buffered_bytes
                    .max(estimated_bytes as u64);
                let write_start = Instant::now();
                let sink_before = sink_metrics.sink_duration();
                writer
                    .write(&batch)
                    .await
                    .map_err(|error| format!("failed writing ordered merge batch: {error}"))?;
                let elapsed = write_start.elapsed();
                let sink_delta = sink_metrics.sink_duration().saturating_sub(sink_before);
                result.write_duration += elapsed;
                result.sink_duration += sink_delta;
                result.encode_duration += elapsed.saturating_sub(sink_delta);
                result.output_batches += 1;
            }

            let close_start = Instant::now();
            let sink_before = sink_metrics.sink_duration();
            writer
                .close()
                .await
                .map_err(|error| format!("failed closing parquet writer: {error}"))?;
            let elapsed = close_start.elapsed();
            let sink_delta = sink_metrics.sink_duration().saturating_sub(sink_before);
            result.write_duration += elapsed;
            result.sink_duration += sink_delta;
            result.close_duration += elapsed;
            Ok(result)
        });

        Self {
            tx: Some(tx),
            handle: Some(handle),
            supports_row_group_copy: false,
        }
    }

    fn new_parallel(
        output_path: PathBuf,
        output_schema: SchemaRef,
        execution_options: ParquetMergeExecutionOptions,
        resolved_parallelism: usize,
        channel_capacity: usize,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<ParquetOutputItem>(channel_capacity.max(1));
        let handle = tokio::spawn(async move {
            run_parallel_output_writer(
                rx,
                output_path,
                output_schema,
                execution_options,
                resolved_parallelism,
            )
            .await
        });

        Self {
            tx: Some(tx),
            handle: Some(handle),
            supports_row_group_copy: true,
        }
    }

    async fn send(&mut self, batch: RecordBatch) -> Result<(), String> {
        self.send_with_memory(batch, OrderedMemoryPermit::none(), 0)
            .await
    }

    async fn send_with_memory(
        &mut self,
        batch: RecordBatch,
        memory_permit: OrderedMemoryPermit,
        estimated_bytes: usize,
    ) -> Result<(), String> {
        let Some(tx) = self.tx.as_ref().cloned() else {
            return Err("ordered writer is no longer accepting batches".to_string());
        };

        if tx
            .send(ParquetOutputItem::Batch {
                batch,
                memory_permit,
                estimated_bytes,
            })
            .await
            .is_err()
        {
            return Err(self
                .take_failure("ordered writer task closed unexpectedly")
                .await);
        }
        Ok(())
    }

    async fn send_row_group_copy_with_memory(
        &mut self,
        copy: RowGroupCopyRequest,
        memory_permit: OrderedMemoryPermit,
        estimated_bytes: usize,
    ) -> Result<(), String> {
        let Some(tx) = self.tx.as_ref().cloned() else {
            return Err("ordered writer is no longer accepting copied row groups".to_string());
        };

        if tx
            .send(ParquetOutputItem::CopyRowGroup {
                copy,
                memory_permit,
                estimated_bytes,
            })
            .await
            .is_err()
        {
            return Err(self
                .take_failure("ordered writer task closed unexpectedly")
                .await);
        }
        Ok(())
    }

    fn supports_row_group_copy(&self) -> bool {
        self.supports_row_group_copy
    }

    async fn finish(&mut self) -> Result<ParquetWriterResult, String> {
        self.tx.take();
        let Some(handle) = self.handle.take() else {
            return Ok(ParquetWriterResult::default());
        };
        match handle.await {
            Ok(result) => result,
            Err(error) => Err(format!("ordered writer task failed to join: {error}")),
        }
    }

    async fn take_failure(&mut self, fallback_message: &str) -> String {
        self.tx.take();
        let Some(handle) = self.handle.take() else {
            return fallback_message.to_string();
        };
        match handle.await {
            Ok(Ok(_)) => fallback_message.to_string(),
            Ok(Err(error)) => error,
            Err(error) => format!("ordered writer task failed to join: {error}"),
        }
    }
}

async fn run_parallel_output_writer(
    mut rx: mpsc::Receiver<ParquetOutputItem>,
    output_path: PathBuf,
    output_schema: SchemaRef,
    execution_options: ParquetMergeExecutionOptions,
    resolved_parallelism: usize,
) -> Result<ParquetWriterResult, String> {
    let mut writer_start = None;
    let (parquet_schema, final_writer_properties) =
        parquet_file_writer_schema_and_properties(output_schema.as_ref(), &execution_options)?;
    let row_group_writer_properties = parquet_writer_properties(&execution_options)?;
    let output_file = StdFile::create(&output_path)
        .map_err(|error| format!("failed to create output parquet file: {error}"))?;
    let mut file_writer = SerializedFileWriter::new(
        output_file,
        parquet_schema,
        Arc::new(final_writer_properties),
    )
    .map_err(|error| format!("failed creating output parquet file writer: {error}"))?;
    let mut accumulator = RowGroupBatchAccumulator::new(
        output_schema,
        execution_options.output_row_group_rows,
        row_group_writer_properties,
    );
    let mut pending = FuturesUnordered::new();
    let mut completed = BTreeMap::new();
    let mut next_to_write = 0_u64;
    let mut result = ParquetWriterResult::default();
    let encoder_limit = resolved_parallelism.max(1);
    let mut writer_buffered_bytes = 0_u64;

    while let Some(item) = rx.recv().await {
        writer_start.get_or_insert_with(Instant::now);
        match item {
            ParquetOutputItem::Batch {
                batch,
                memory_permit,
                estimated_bytes,
            } => {
                result.output_batches += 1;
                let force_partial_row_group = execution_options
                    .ordered_memory_budget_bytes
                    .map(|budget| estimated_bytes >= budget.saturating_div(2).max(1))
                    .unwrap_or(false);
                writer_buffered_bytes =
                    writer_buffered_bytes.saturating_add(estimated_bytes as u64);
                result.writer_peak_buffered_bytes =
                    result.writer_peak_buffered_bytes.max(writer_buffered_bytes);
                for job in accumulator.push(batch, memory_permit) {
                    while pending.len() >= encoder_limit {
                        drain_one_encoded_row_group(
                            &mut pending,
                            &mut completed,
                            &mut file_writer,
                            &mut next_to_write,
                            &mut result,
                            &mut writer_buffered_bytes,
                        )
                        .await?;
                    }
                    spawn_row_group_encoder(&mut pending, job);
                }
                if force_partial_row_group {
                    if let Some(job) = accumulator.finish() {
                        while pending.len() >= encoder_limit {
                            drain_one_encoded_row_group(
                                &mut pending,
                                &mut completed,
                                &mut file_writer,
                                &mut next_to_write,
                                &mut result,
                                &mut writer_buffered_bytes,
                            )
                            .await?;
                        }
                        spawn_row_group_encoder(&mut pending, job);
                    }
                }
                if let Some(budget) = execution_options.ordered_memory_budget_bytes {
                    let pressure_bytes = budget.saturating_div(2).max(1) as u64;
                    while writer_buffered_bytes >= pressure_bytes && !pending.is_empty() {
                        drain_one_encoded_row_group(
                            &mut pending,
                            &mut completed,
                            &mut file_writer,
                            &mut next_to_write,
                            &mut result,
                            &mut writer_buffered_bytes,
                        )
                        .await?;
                    }
                }
            }
            ParquetOutputItem::CopyRowGroup {
                copy,
                memory_permit,
                estimated_bytes,
            } => {
                writer_buffered_bytes =
                    writer_buffered_bytes.saturating_add(estimated_bytes as u64);
                result.writer_peak_buffered_bytes =
                    result.writer_peak_buffered_bytes.max(writer_buffered_bytes);
                if let Some(job) = accumulator.finish() {
                    while pending.len() >= encoder_limit {
                        drain_one_encoded_row_group(
                            &mut pending,
                            &mut completed,
                            &mut file_writer,
                            &mut next_to_write,
                            &mut result,
                            &mut writer_buffered_bytes,
                        )
                        .await?;
                    }
                    spawn_row_group_encoder(&mut pending, job);
                }
                while pending.len() >= encoder_limit {
                    drain_one_encoded_row_group(
                        &mut pending,
                        &mut completed,
                        &mut file_writer,
                        &mut next_to_write,
                        &mut result,
                        &mut writer_buffered_bytes,
                    )
                    .await?;
                }
                let sequence = accumulator.next_sequence();
                let encoded = EncodedRowGroup {
                    sequence,
                    data: EncodedRowGroupData::Copied(copy.clone()),
                    row_group_metadata: copy.row_group_metadata.clone(),
                    encode_duration: Duration::default(),
                    estimated_bytes,
                    _memory_permits: vec![memory_permit],
                };
                completed.insert(sequence, encoded);
                write_ready_encoded_row_groups(
                    &mut completed,
                    &mut file_writer,
                    &mut next_to_write,
                    &mut result,
                    &mut writer_buffered_bytes,
                )?;
            }
        }
    }

    if let Some(job) = accumulator.finish() {
        while pending.len() >= encoder_limit {
            drain_one_encoded_row_group(
                &mut pending,
                &mut completed,
                &mut file_writer,
                &mut next_to_write,
                &mut result,
                &mut writer_buffered_bytes,
            )
            .await?;
        }
        spawn_row_group_encoder(&mut pending, job);
    }

    while !pending.is_empty() {
        drain_one_encoded_row_group(
            &mut pending,
            &mut completed,
            &mut file_writer,
            &mut next_to_write,
            &mut result,
            &mut writer_buffered_bytes,
        )
        .await?;
    }
    write_ready_encoded_row_groups(
        &mut completed,
        &mut file_writer,
        &mut next_to_write,
        &mut result,
        &mut writer_buffered_bytes,
    )?;
    if !completed.is_empty() {
        return Err("encoded row group assembly ended with out-of-order gaps".to_string());
    }

    let close_start = Instant::now();
    file_writer
        .close()
        .map_err(|error| format!("failed closing parquet writer: {error}"))?;
    result.close_duration += close_start.elapsed();
    result.write_duration = writer_start
        .map(|start| start.elapsed())
        .unwrap_or(result.close_duration);
    Ok(result)
}

async fn drain_one_encoded_row_group<W: Write + Send>(
    pending: &mut FuturesUnordered<RowGroupEncodeHandle>,
    completed: &mut BTreeMap<u64, EncodedRowGroup>,
    writer: &mut SerializedFileWriter<W>,
    next_to_write: &mut u64,
    result: &mut ParquetWriterResult,
    writer_buffered_bytes: &mut u64,
) -> Result<(), String> {
    let Some(join_result) = pending.next().await else {
        return Ok(());
    };
    let encoded = match join_result {
        Ok(Ok(encoded)) => encoded,
        Ok(Err(error)) => return Err(error),
        Err(error) => return Err(format!("row group encoder task failed to join: {error}")),
    };
    result.encode_duration += encoded.encode_duration;
    completed.insert(encoded.sequence, encoded);
    write_ready_encoded_row_groups(
        completed,
        writer,
        next_to_write,
        result,
        writer_buffered_bytes,
    )
}

fn write_ready_encoded_row_groups<W: Write + Send>(
    completed: &mut BTreeMap<u64, EncodedRowGroup>,
    writer: &mut SerializedFileWriter<W>,
    next_to_write: &mut u64,
    result: &mut ParquetWriterResult,
    writer_buffered_bytes: &mut u64,
) -> Result<(), String> {
    while let Some(encoded) = completed.remove(&*next_to_write) {
        let estimated_bytes = encoded.estimated_bytes as u64;
        let copied_metadata = match &encoded.data {
            EncodedRowGroupData::Copied(copy) => Some((copy.rows, copy.compressed_bytes)),
            EncodedRowGroupData::InMemory(_) => None,
        };
        let sink_start = Instant::now();
        let copied = append_encoded_row_group(writer, encoded)?;
        let elapsed = sink_start.elapsed();
        result.sink_duration += elapsed;
        if copied {
            let (rows, compressed_bytes) = copied_metadata
                .ok_or_else(|| "copied row group missing copy metadata".to_string())?;
            result.copied_row_groups += 1;
            result.copied_rows += rows;
            result.copied_compressed_bytes += compressed_bytes;
            result.row_group_copy_duration += elapsed;
        }
        *writer_buffered_bytes = writer_buffered_bytes.saturating_sub(estimated_bytes);
        *next_to_write += 1;
    }
    Ok(())
}

#[cfg(test)]
#[allow(dead_code)]
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

#[cfg(test)]
#[allow(dead_code)]
impl PreparedOrderBatch {
    fn new(
        batch: RecordBatch,
        order_plan: &ParquetOrderPlan,
        row_group_index: usize,
        row_group_batch_index: usize,
    ) -> Result<Self, String> {
        let order_column = extract_prepared_order_column(&batch, order_plan)?;
        let non_null_prefix_len = order_column.non_null_prefix_len(batch.num_rows());
        Ok(Self {
            batch,
            order_column,
            row_group_index,
            row_group_batch_index,
            non_null_prefix_len,
        })
    }
}

#[cfg(test)]
#[allow(dead_code)]
impl PreparedOrderColumn {
    fn is_null(&self, row: usize) -> bool {
        match self {
            Self::Int64(array) => array.is_null(row),
            Self::UInt64(array) => array.is_null(row),
            Self::Float64(array) => array.is_null(row),
            Self::Utf8(array) => array.is_null(row),
            Self::LargeUtf8(array) => array.is_null(row),
            Self::Date32(array) => array.is_null(row),
            Self::Date64(array) => array.is_null(row),
            Self::TimestampSecond(array) => array.is_null(row),
            Self::TimestampMillisecond(array) => array.is_null(row),
            Self::TimestampMicrosecond(array) => array.is_null(row),
            Self::TimestampNanosecond(array) => array.is_null(row),
        }
    }

    fn non_null_prefix_len(&self, row_count: usize) -> usize {
        let mut low = 0;
        let mut high = row_count;
        while low < high {
            let mid = low + ((high - low) / 2);
            if self.is_null(mid) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        low
    }

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

    fn compare_row_to_other(
        &self,
        row: usize,
        other: &PreparedOrderColumn,
        other_row: usize,
    ) -> CmpOrdering {
        match (self, other) {
            (Self::Int64(left), Self::Int64(right)) => {
                compare_int64_arrays(left, row, right, other_row)
            }
            (Self::UInt64(left), Self::UInt64(right)) => {
                compare_uint64_arrays(left, row, right, other_row)
            }
            (Self::Float64(left), Self::Float64(right)) => {
                compare_float64_arrays(left, row, right, other_row)
            }
            (Self::Utf8(left), Self::Utf8(right)) => {
                compare_string_arrays(left, row, right, other_row)
            }
            (Self::LargeUtf8(left), Self::LargeUtf8(right)) => {
                compare_large_string_arrays(left, row, right, other_row)
            }
            (Self::Date32(left), Self::Date32(right)) => {
                compare_date32_arrays(left, row, right, other_row)
            }
            (Self::Date64(left), Self::Date64(right)) => {
                compare_date64_arrays(left, row, right, other_row)
            }
            (Self::TimestampSecond(left), Self::TimestampSecond(right)) => {
                compare_timestamp_second_arrays(left, row, right, other_row)
            }
            (Self::TimestampMillisecond(left), Self::TimestampMillisecond(right)) => {
                compare_timestamp_millisecond_arrays(left, row, right, other_row)
            }
            (Self::TimestampMicrosecond(left), Self::TimestampMicrosecond(right)) => {
                compare_timestamp_microsecond_arrays(left, row, right, other_row)
            }
            (Self::TimestampNanosecond(left), Self::TimestampNanosecond(right)) => {
                compare_timestamp_nanosecond_arrays(left, row, right, other_row)
            }
            _ => CmpOrdering::Equal,
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

#[cfg(test)]
#[allow(dead_code)]
fn compare_source_heads(
    sources: &[OrderedMergeSource],
    left_index: usize,
    right_index: usize,
) -> CmpOrdering {
    sources[left_index].compare_head_to_source(&sources[right_index])
}

#[cfg(test)]
fn row_group_fast_path_safe(
    candidate_index: usize,
    sources: &[OrderedMergeSource],
    source_order_metadata: &[SourceOrderMetadata],
) -> bool {
    let Some(source) = sources.get(candidate_index) else {
        return false;
    };
    let Some(current_batch) = source.current_batch.as_ref() else {
        return false;
    };
    let Some(range) = source_order_metadata
        .get(candidate_index)
        .and_then(|metadata| metadata.row_groups.get(current_batch.row_group_index))
    else {
        return false;
    };

    match range.null_kind {
        RowGroupNullKind::MixedOrUnknown => false,
        RowGroupNullKind::NoNulls => {
            let (Some(min), Some(max)) = (range.min.as_ref(), range.max.as_ref()) else {
                return false;
            };
            if compare_order_key_values(min, max) == CmpOrdering::Greater {
                return false;
            }
            for (competitor_index, competitor) in sources.iter().enumerate() {
                if competitor_index == candidate_index || competitor.current_batch.is_none() {
                    continue;
                }
                let ordering = compare_value_to_source_head(max, competitor);
                if ordering == CmpOrdering::Greater
                    || (ordering == CmpOrdering::Equal && candidate_index > competitor_index)
                {
                    return false;
                }
            }
            true
        }
        RowGroupNullKind::AllNulls => {
            for (competitor_index, competitor) in sources.iter().enumerate() {
                if competitor_index == candidate_index || competitor.current_batch.is_none() {
                    continue;
                }
                if competitor.has_non_null_remaining() || candidate_index > competitor_index {
                    return false;
                }
            }
            true
        }
    }
}

#[cfg(test)]
fn compare_value_to_source_head(value: &OrderKeyValue, source: &OrderedMergeSource) -> CmpOrdering {
    let batch = source
        .current_batch
        .as_ref()
        .expect("source head comparison requires an active batch");
    batch
        .order_column
        .compare_row_to_value(source.current_row, value)
        .reverse()
}

async fn merge_parquet_files_unordered(
    builders: Vec<ParquetRecordBatchStreamBuilder<File>>,
    output_path: &Path,
    output_schema: SchemaRef,
    source_adapters: Vec<Arc<CompiledSourceAdapter>>,
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<ParquetMergeRunStats, String> {
    let resolved_parallelism = resolve_parquet_parallelism(execution_options, builders.len());
    if resolved_parallelism <= 1 {
        return merge_parquet_files_unordered_serial(
            builders,
            output_path,
            output_schema,
            source_adapters,
            execution_options,
        )
        .await;
    }

    match execution_options.unordered_merge_order {
        UnorderedMergeOrder::PreserveInputOrder => {
            merge_parquet_files_unordered_preserve_order(
                builders,
                output_path,
                output_schema,
                source_adapters,
                execution_options,
                resolved_parallelism,
            )
            .await
        }
        UnorderedMergeOrder::AllowInterleaved => {
            merge_parquet_files_unordered_interleaved(
                builders,
                output_path,
                output_schema,
                source_adapters,
                execution_options,
                resolved_parallelism,
            )
            .await
        }
    }
}

async fn create_async_parquet_writer(
    output_path: &Path,
    output_schema: SchemaRef,
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<(TimedAsyncArrowWriter, Arc<ParquetAsyncSinkMetrics>), String> {
    let output_file = File::create(output_path)
        .await
        .map_err(|error| format!("failed to create output parquet file: {error}"))?;
    let output_file = BufWriter::with_capacity(1 << 20, output_file);
    let sink_metrics = Arc::new(ParquetAsyncSinkMetrics::default());
    let output_file = TimedAsyncFileWriter::new(output_file, sink_metrics.clone());
    let writer_properties = parquet_writer_properties(execution_options)?;
    let writer = AsyncArrowWriter::try_new(output_file, output_schema, Some(writer_properties))
        .map_err(|error| format!("failed to create parquet writer: {error}"))?;
    Ok((writer, sink_metrics))
}

async fn create_parquet_output_writer(
    output_path: &Path,
    output_schema: SchemaRef,
    execution_options: &ParquetMergeExecutionOptions,
    resolved_parallelism: usize,
    channel_capacity: usize,
) -> Result<ParquetOutputWriter, String> {
    if resolved_parallelism <= 1 {
        let (writer, sink_metrics) =
            create_async_parquet_writer(output_path, output_schema, execution_options).await?;
        Ok(ParquetOutputWriter::new_serial(
            writer,
            sink_metrics,
            channel_capacity,
        ))
    } else {
        Ok(ParquetOutputWriter::new_parallel(
            output_path.to_path_buf(),
            output_schema,
            execution_options.clone(),
            resolved_parallelism,
            channel_capacity,
        ))
    }
}

async fn merge_parquet_files_unordered_serial(
    builders: Vec<ParquetRecordBatchStreamBuilder<File>>,
    output_path: &Path,
    output_schema: SchemaRef,
    source_adapters: Vec<Arc<CompiledSourceAdapter>>,
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<ParquetMergeRunStats, String> {
    let execution_start = Instant::now();
    let (mut writer, sink_metrics) =
        create_async_parquet_writer(output_path, output_schema.clone(), execution_options).await?;

    let mut rows = 0_u64;
    let mut input_batches = 0_u64;
    let mut output_batches = 0_u64;
    let mut read_decode_duration = Duration::default();
    let mut source_prepare_duration = Duration::default();
    let mut writer_write_duration = Duration::default();
    let mut writer_encode_duration = Duration::default();
    let mut writer_sink_duration = Duration::default();
    let mut writer_close_duration = Duration::default();

    for (builder, adapter) in builders.into_iter().zip(source_adapters.into_iter()) {
        let mut scratch = ExecutionScratch::default();
        let mut stream = builder
            .with_batch_size(execution_options.read_batch_size)
            .build()
            .map_err(|error| format!("failed to build parquet stream: {error}"))?;
        loop {
            let decode_start = Instant::now();
            let batch_result = stream.next().await;
            read_decode_duration += decode_start.elapsed();
            let Some(batch_result) = batch_result else {
                break;
            };
            let batch =
                batch_result.map_err(|error| format!("failed reading parquet batch: {error}"))?;
            input_batches += 1;
            rows += batch.num_rows() as u64;
            let prepare_start = Instant::now();
            let adjusted_batch = adapter
                .adapt_batch(&batch, &output_schema, &mut scratch)
                .map_err(|error| format!("failed adapting parquet batch: {error}"))?;
            source_prepare_duration += prepare_start.elapsed();
            let write_start = Instant::now();
            let sink_before = sink_metrics.sink_duration();
            writer
                .write(&adjusted_batch)
                .await
                .map_err(|error| format!("failed writing merged parquet batch: {error}"))?;
            let elapsed = write_start.elapsed();
            let sink_delta = sink_metrics.sink_duration().saturating_sub(sink_before);
            writer_write_duration += elapsed;
            writer_sink_duration += sink_delta;
            writer_encode_duration += elapsed.saturating_sub(sink_delta);
            output_batches += 1;
        }
    }

    let close_start = Instant::now();
    let sink_before = sink_metrics.sink_duration();
    writer
        .close()
        .await
        .map_err(|error| format!("failed closing parquet writer: {error}"))?;
    let elapsed = close_start.elapsed();
    let sink_delta = sink_metrics.sink_duration().saturating_sub(sink_before);
    writer_write_duration += elapsed;
    writer_sink_duration += sink_delta;
    writer_close_duration += elapsed;

    Ok(ParquetMergeRunStats {
        rows,
        input_batches,
        output_batches,
        execution_duration: execution_start.elapsed(),
        read_decode_duration,
        source_prepare_duration,
        writer_write_duration,
        writer_encode_duration,
        writer_sink_duration,
        writer_close_duration,
        ..ParquetMergeRunStats::default()
    })
}

#[derive(Debug)]
struct UnorderedPreparedBatch {
    batch: RecordBatch,
    rows: usize,
}

async fn merge_parquet_files_unordered_preserve_order(
    builders: Vec<ParquetRecordBatchStreamBuilder<File>>,
    output_path: &Path,
    output_schema: SchemaRef,
    source_adapters: Vec<Arc<CompiledSourceAdapter>>,
    execution_options: &ParquetMergeExecutionOptions,
    resolved_parallelism: usize,
) -> Result<ParquetMergeRunStats, String> {
    let execution_start = Instant::now();
    let mut output_writer = create_parquet_output_writer(
        output_path,
        output_schema.clone(),
        execution_options,
        resolved_parallelism,
        execution_options.prefetch_batches_per_source.max(1) * resolved_parallelism.max(1),
    )
    .await?;
    let semaphore = Arc::new(Semaphore::new(resolved_parallelism.max(1)));
    let worker_metrics = Arc::new(ParquetWorkerMetrics::default());
    let mut receivers = Vec::with_capacity(builders.len());
    let mut worker_handles = Vec::with_capacity(builders.len());

    for (source_index, (builder, adapter)) in builders
        .into_iter()
        .zip(source_adapters.into_iter())
        .enumerate()
    {
        let (tx, rx) = mpsc::channel(execution_options.prefetch_batches_per_source);
        let output_schema = output_schema.clone();
        let execution_options = execution_options.clone();
        let semaphore = semaphore.clone();
        let worker_metrics = worker_metrics.clone();
        let handle = tokio::spawn(async move {
            parquet_unordered_source_worker(
                source_index,
                builder,
                output_schema,
                adapter,
                execution_options,
                semaphore,
                worker_metrics,
                tx,
            )
            .await
        });
        receivers.push(rx);
        worker_handles.push(handle);
    }

    let mut stats = ParquetMergeRunStats::default();
    let mut first_error = None;
    for receiver in &mut receivers {
        while let Some(message) = receiver.recv().await {
            match message {
                Ok(prepared) => {
                    stats.input_batches += 1;
                    stats.rows += prepared.rows as u64;
                    if let Err(error) = output_writer.send(prepared.batch).await {
                        first_error = Some(error);
                        break;
                    }
                }
                Err(error) => {
                    first_error = Some(error);
                    break;
                }
            }
        }
        if first_error.is_some() {
            break;
        }
    }
    drop(receivers);

    for handle in worker_handles {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                first_error.get_or_insert(error);
            }
            Err(error) => {
                first_error
                    .get_or_insert(format!("unordered merge worker failed to join: {error}"));
            }
        };
    }

    if let Some(error) = first_error {
        let _ = output_writer.finish().await;
        return Err(error);
    }

    let writer_result = output_writer.finish().await?;
    stats.output_batches = writer_result.output_batches;
    stats.writer_write_duration = writer_result.write_duration;
    stats.writer_encode_duration = writer_result.encode_duration;
    stats.writer_sink_duration = writer_result.sink_duration;
    stats.writer_close_duration = writer_result.close_duration;
    stats.copied_row_groups = writer_result.copied_row_groups;
    stats.copied_rows = writer_result.copied_rows;
    stats.copied_compressed_bytes = writer_result.copied_compressed_bytes;
    stats.row_group_copy_duration = writer_result.row_group_copy_duration;
    stats.writer_peak_buffered_bytes = writer_result.writer_peak_buffered_bytes;
    stats.read_decode_duration =
        Duration::from_nanos(worker_metrics.read_decode_nanos.load(Ordering::Relaxed));
    stats.source_prepare_duration =
        Duration::from_nanos(worker_metrics.source_prepare_nanos.load(Ordering::Relaxed));
    stats.execution_duration = execution_start.elapsed();
    Ok(stats)
}

async fn merge_parquet_files_unordered_interleaved(
    builders: Vec<ParquetRecordBatchStreamBuilder<File>>,
    output_path: &Path,
    output_schema: SchemaRef,
    source_adapters: Vec<Arc<CompiledSourceAdapter>>,
    execution_options: &ParquetMergeExecutionOptions,
    resolved_parallelism: usize,
) -> Result<ParquetMergeRunStats, String> {
    let execution_start = Instant::now();
    let mut output_writer = create_parquet_output_writer(
        output_path,
        output_schema.clone(),
        execution_options,
        resolved_parallelism,
        execution_options.prefetch_batches_per_source.max(1) * resolved_parallelism.max(1),
    )
    .await?;
    let semaphore = Arc::new(Semaphore::new(resolved_parallelism.max(1)));
    let worker_metrics = Arc::new(ParquetWorkerMetrics::default());
    let channel_capacity =
        execution_options.prefetch_batches_per_source.max(1) * builders.len().max(1);
    let (tx, mut rx) = mpsc::channel(channel_capacity);
    let mut worker_handles = Vec::with_capacity(builders.len());

    for (source_index, (builder, adapter)) in builders
        .into_iter()
        .zip(source_adapters.into_iter())
        .enumerate()
    {
        let tx = tx.clone();
        let output_schema = output_schema.clone();
        let execution_options = execution_options.clone();
        let semaphore = semaphore.clone();
        let worker_metrics = worker_metrics.clone();
        let handle = tokio::spawn(async move {
            parquet_unordered_source_worker(
                source_index,
                builder,
                output_schema,
                adapter,
                execution_options,
                semaphore,
                worker_metrics,
                tx,
            )
            .await
        });
        worker_handles.push(handle);
    }
    drop(tx);

    let mut stats = ParquetMergeRunStats::default();
    let mut first_error = None;
    while let Some(message) = rx.recv().await {
        match message {
            Ok(prepared) => {
                stats.input_batches += 1;
                stats.rows += prepared.rows as u64;
                if let Err(error) = output_writer.send(prepared.batch).await {
                    first_error = Some(error);
                    break;
                }
            }
            Err(error) => {
                first_error = Some(error);
                break;
            }
        }
    }
    drop(rx);

    for handle in worker_handles {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                first_error.get_or_insert(error);
            }
            Err(error) => {
                first_error
                    .get_or_insert(format!("unordered merge worker failed to join: {error}"));
            }
        };
    }

    if let Some(error) = first_error {
        let _ = output_writer.finish().await;
        return Err(error);
    }

    let writer_result = output_writer.finish().await?;
    stats.output_batches = writer_result.output_batches;
    stats.writer_write_duration = writer_result.write_duration;
    stats.writer_encode_duration = writer_result.encode_duration;
    stats.writer_sink_duration = writer_result.sink_duration;
    stats.writer_close_duration = writer_result.close_duration;
    stats.copied_row_groups = writer_result.copied_row_groups;
    stats.copied_rows = writer_result.copied_rows;
    stats.copied_compressed_bytes = writer_result.copied_compressed_bytes;
    stats.row_group_copy_duration = writer_result.row_group_copy_duration;
    stats.writer_peak_buffered_bytes = writer_result.writer_peak_buffered_bytes;
    stats.read_decode_duration =
        Duration::from_nanos(worker_metrics.read_decode_nanos.load(Ordering::Relaxed));
    stats.source_prepare_duration =
        Duration::from_nanos(worker_metrics.source_prepare_nanos.load(Ordering::Relaxed));
    stats.execution_duration = execution_start.elapsed();
    Ok(stats)
}

async fn parquet_unordered_source_worker(
    source_index: usize,
    builder: ParquetRecordBatchStreamBuilder<File>,
    output_schema: SchemaRef,
    adapter: Arc<CompiledSourceAdapter>,
    execution_options: ParquetMergeExecutionOptions,
    semaphore: Arc<Semaphore>,
    worker_metrics: Arc<ParquetWorkerMetrics>,
    tx: mpsc::Sender<Result<UnorderedPreparedBatch, String>>,
) -> Result<(), String> {
    let result: Result<(), String> = async {
        let mut scratch = ExecutionScratch::default();
        let mut stream = builder
            .with_batch_size(execution_options.read_batch_size)
            .build()
            .map_err(|error| {
                format!("failed to build parquet stream for source {source_index}: {error}")
            })?;

        loop {
            let batch_result = {
                let _permit = semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| "parquet merge parallelism semaphore closed".to_string())?;
                let decode_start = Instant::now();
                let batch_result = stream.next().await;
                worker_metrics
                    .read_decode_nanos
                    .fetch_add(decode_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                batch_result
            };

            let Some(batch_result) = batch_result else {
                break;
            };
            let batch = batch_result.map_err(|error| {
                format!("failed reading parquet batch from source {source_index}: {error}")
            })?;
            let rows = batch.num_rows();
            let prepared = {
                let _permit = semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| "parquet merge parallelism semaphore closed".to_string())?;
                let prepare_start = Instant::now();
                let adapted = adapter
                    .adapt_batch(&batch, &output_schema, &mut scratch)
                    .map_err(|error| {
                        format!("failed adapting parquet batch from source {source_index}: {error}")
                    })?;
                worker_metrics
                    .source_prepare_nanos
                    .fetch_add(prepare_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                UnorderedPreparedBatch {
                    batch: adapted,
                    rows,
                }
            };

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

async fn merge_payload_parquet_files_ordered(
    input_paths: &[PathBuf],
    output_path: &Path,
    output_schema: SchemaRef,
    source_schemas: Vec<SchemaRef>,
    source_adapters: Vec<Arc<CompiledSourceAdapter>>,
    source_order_metadata: Vec<SourceOrderMetadata>,
    order_plan: ParquetOrderPlan,
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<ParquetMergeRunStats, String> {
    match order_plan.key_type {
        ParquetOrderKeyType::Int64 => {
            merge_payload_parquet_files_ordered_typed::<Int64OrderKey>(
                input_paths,
                output_path,
                output_schema,
                source_schemas,
                source_adapters,
                source_order_metadata,
                order_plan,
                execution_options,
            )
            .await
        }
        ParquetOrderKeyType::UInt64 => {
            merge_payload_parquet_files_ordered_typed::<UInt64OrderKey>(
                input_paths,
                output_path,
                output_schema,
                source_schemas,
                source_adapters,
                source_order_metadata,
                order_plan,
                execution_options,
            )
            .await
        }
        ParquetOrderKeyType::Float64 => {
            merge_payload_parquet_files_ordered_typed::<Float64OrderKey>(
                input_paths,
                output_path,
                output_schema,
                source_schemas,
                source_adapters,
                source_order_metadata,
                order_plan,
                execution_options,
            )
            .await
        }
        ParquetOrderKeyType::Utf8 => {
            merge_payload_parquet_files_ordered_typed::<Utf8OrderKey>(
                input_paths,
                output_path,
                output_schema,
                source_schemas,
                source_adapters,
                source_order_metadata,
                order_plan,
                execution_options,
            )
            .await
        }
        ParquetOrderKeyType::LargeUtf8 => {
            merge_payload_parquet_files_ordered_typed::<LargeUtf8OrderKey>(
                input_paths,
                output_path,
                output_schema,
                source_schemas,
                source_adapters,
                source_order_metadata,
                order_plan,
                execution_options,
            )
            .await
        }
        ParquetOrderKeyType::Date32 => {
            merge_payload_parquet_files_ordered_typed::<Date32OrderKey>(
                input_paths,
                output_path,
                output_schema,
                source_schemas,
                source_adapters,
                source_order_metadata,
                order_plan,
                execution_options,
            )
            .await
        }
        ParquetOrderKeyType::Date64 => {
            merge_payload_parquet_files_ordered_typed::<Date64OrderKey>(
                input_paths,
                output_path,
                output_schema,
                source_schemas,
                source_adapters,
                source_order_metadata,
                order_plan,
                execution_options,
            )
            .await
        }
        ParquetOrderKeyType::TimestampSecond => {
            merge_payload_parquet_files_ordered_typed::<TimestampSecondOrderKey>(
                input_paths,
                output_path,
                output_schema,
                source_schemas,
                source_adapters,
                source_order_metadata,
                order_plan,
                execution_options,
            )
            .await
        }
        ParquetOrderKeyType::TimestampMillisecond => {
            merge_payload_parquet_files_ordered_typed::<TimestampMillisecondOrderKey>(
                input_paths,
                output_path,
                output_schema,
                source_schemas,
                source_adapters,
                source_order_metadata,
                order_plan,
                execution_options,
            )
            .await
        }
        ParquetOrderKeyType::TimestampMicrosecond => {
            merge_payload_parquet_files_ordered_typed::<TimestampMicrosecondOrderKey>(
                input_paths,
                output_path,
                output_schema,
                source_schemas,
                source_adapters,
                source_order_metadata,
                order_plan,
                execution_options,
            )
            .await
        }
        ParquetOrderKeyType::TimestampNanosecond => {
            merge_payload_parquet_files_ordered_typed::<TimestampNanosecondOrderKey>(
                input_paths,
                output_path,
                output_schema,
                source_schemas,
                source_adapters,
                source_order_metadata,
                order_plan,
                execution_options,
            )
            .await
        }
    }
}

async fn merge_payload_parquet_files_ordered_typed<K: OrderedKey>(
    input_paths: &[PathBuf],
    output_path: &Path,
    output_schema: SchemaRef,
    source_schemas: Vec<SchemaRef>,
    source_adapters: Vec<Arc<CompiledSourceAdapter>>,
    source_order_metadata: Vec<SourceOrderMetadata>,
    order_plan: ParquetOrderPlan,
    execution_options: &ParquetMergeExecutionOptions,
) -> Result<ParquetMergeRunStats, String> {
    debug_assert_eq!(order_plan.key_type, K::KEY_TYPE);
    let execution_start = Instant::now();
    let order_plan = Arc::new(order_plan);
    let worker_metrics = Arc::new(ParquetWorkerMetrics::default());
    let source_parallelism = resolve_parquet_parallelism(execution_options, input_paths.len());
    let cpu_parallelism = resolve_parquet_cpu_parallelism(execution_options);
    let ordered_memory_limiter = execution_options
        .ordered_memory_budget_bytes
        .map(OrderedMemoryLimiter::new);
    let mut writer_execution_options = execution_options.clone();
    if matches!(
        K::KEY_TYPE,
        ParquetOrderKeyType::Utf8 | ParquetOrderKeyType::LargeUtf8
    ) && execution_options.ordered_memory_budget_bytes.is_some()
    {
        writer_execution_options.output_batch_rows = writer_execution_options
            .output_batch_rows
            .max(writer_execution_options.output_row_group_rows);
    }
    let mut ordered_writer = create_parquet_output_writer(
        output_path,
        output_schema.clone(),
        &writer_execution_options,
        cpu_parallelism,
        execution_options.prefetch_batches_per_source.max(2) * cpu_parallelism.max(1),
    )
    .await?;
    let pipeline_limit = if execution_options.ordered_memory_budget_bytes.is_some() {
        cpu_parallelism.saturating_mul(4).max(1)
    } else {
        source_parallelism.max(1)
    };
    let mut ordered_output_pipeline =
        OrderedOutputPipeline::new(pipeline_limit, ordered_memory_limiter.clone());
    let semaphore = Arc::new(Semaphore::new(source_parallelism.max(1)));
    let source_copy_plans = if execution_options.stats_fast_path {
        build_ordered_row_group_copy_plans(
            input_paths,
            &source_schemas,
            output_schema.as_ref(),
            &source_adapters,
            execution_options,
        )?
    } else {
        vec![OrderedRowGroupCopyPlan::default(); input_paths.len()]
    };
    let mut receivers = Vec::with_capacity(input_paths.len());
    let mut worker_handles = Vec::with_capacity(input_paths.len());
    for (source_index, (input_path, adapter)) in input_paths
        .iter()
        .cloned()
        .zip(source_adapters.into_iter())
        .enumerate()
    {
        let (tx, rx) = mpsc::channel(execution_options.prefetch_batches_per_source);
        let (command_tx, command_rx) = mpsc::channel(1);
        let output_schema = output_schema.clone();
        let order_plan = order_plan.clone();
        let execution_options = execution_options.clone();
        let worker_metrics = worker_metrics.clone();
        let semaphore = semaphore.clone();
        let source_order_metadata = source_order_metadata
            .get(source_index)
            .cloned()
            .unwrap_or_default();
        let source_copy_plan = source_copy_plans
            .get(source_index)
            .cloned()
            .unwrap_or_default();
        let handle = tokio::spawn(async move {
            parquet_merge_source_worker_typed::<K>(
                source_index,
                input_path,
                output_schema,
                adapter,
                order_plan,
                execution_options,
                semaphore,
                worker_metrics,
                source_order_metadata,
                source_copy_plan,
                command_rx,
                tx,
            )
            .await
        });
        receivers.push((rx, command_tx));
        worker_handles.push(handle);
    }

    let ordered_merge_start = Instant::now();
    let mut stats = ParquetMergeRunStats::default();
    let mut accumulator = OrderedOutputAccumulator::new(output_schema.clone());
    let mut sources = receivers
        .into_iter()
        .enumerate()
        .map(|(source_index, (receiver, command_tx))| {
            OrderedMergeSourceTyped::<K>::new(source_index, receiver, command_tx)
        })
        .collect::<Vec<_>>();
    let mut selector = TypedSourceTournament::<K>::new(sources.len());

    for source_index in 0..sources.len() {
        if sources[source_index]
            .load_next(&mut stats.input_batches)
            .await?
            && sources[source_index].current_batch.is_some()
        {
            selector.set_active(source_index, true, &sources);
        }
    }

    while sources.iter().any(OrderedMergeSourceTyped::has_active_head) {
        if execution_options.stats_fast_path {
            if let Some(source_index) =
                select_fast_path_source_typed::<K>(&sources, &source_order_metadata)
            {
                if sources[source_index].current_batch.is_some() {
                    selector.remove(source_index, &sources);
                }
                let fast_path_start = Instant::now();
                let (row_groups, batches) = drain_fast_path_source_typed::<K>(
                    source_index,
                    &mut sources,
                    &source_order_metadata,
                    &source_copy_plans,
                    &mut ordered_writer,
                    &mut ordered_output_pipeline,
                    execution_options,
                    &mut accumulator,
                    &mut stats,
                )
                .await?;
                stats.stats_fast_path_duration += fast_path_start.elapsed();
                stats.fast_path_row_groups += row_groups;
                stats.fast_path_batches += batches;
                if sources[source_index].current_batch.is_some() {
                    selector.set_active(source_index, true, &sources);
                } else {
                    selector.set_active(source_index, false, &sources);
                }
                stats.ordered_selector_comparisons += selector.take_comparisons();
                continue;
            }
        }

        if decode_pending_copy_candidates_typed::<K>(&mut sources, &mut selector, &mut stats)
            .await?
        {
            stats.ordered_selector_comparisons += selector.take_comparisons();
            continue;
        }

        if try_dense_ordered_merge_window_typed::<K>(
            &mut sources,
            &mut selector,
            &mut ordered_writer,
            &mut ordered_output_pipeline,
            execution_options,
            &mut accumulator,
            &mut stats,
            cpu_parallelism,
            order_plan.as_ref(),
        )
        .await?
        {
            continue;
        }

        if selector.is_empty() {
            break;
        }

        let selection_start = Instant::now();
        let entry = selector
            .winner()
            .expect("non-empty ordered merge selector always has a winner");
        let next_competitor = selector
            .next_competitor(entry.source_index, &sources)
            .map(|entry| entry.source_index);
        let counts_as_fallback_batch = sources[entry.source_index].current_row == 0;
        let run_len = {
            let source = &sources[entry.source_index];
            let competitor = next_competitor.map(|index| &sources[index]);
            source.contiguous_run_len(
                competitor,
                execution_options
                    .output_batch_rows
                    .saturating_sub(accumulator.rows())
                    .max(1),
            )
        };
        add_ordered_selection_duration(&mut stats, selection_start.elapsed());
        append_rows_from_source_typed::<K>(
            entry.source_index,
            run_len,
            &mut sources,
            &mut ordered_writer,
            &mut ordered_output_pipeline,
            execution_options,
            &mut accumulator,
            &mut stats,
        )
        .await?;
        if counts_as_fallback_batch {
            stats.fallback_batches += 1;
        }

        if sources[entry.source_index].current_batch.is_some() {
            selector.set_active(entry.source_index, true, &sources);
        } else {
            selector.set_active(entry.source_index, false, &sources);
        }
        stats.ordered_selector_comparisons += selector.take_comparisons();
    }

    flush_ordered_accumulator(
        &mut accumulator,
        &mut ordered_output_pipeline,
        &mut ordered_writer,
        &mut stats,
    )
    .await?;
    ordered_output_pipeline
        .finish(&mut ordered_writer, &mut stats)
        .await?;

    stats.ordered_merge_duration = ordered_merge_start.elapsed();

    for handle in worker_handles {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => return Err(error),
            Err(error) => return Err(format!("ordered merge worker failed to join: {error}")),
        }
    }

    let writer_result = ordered_writer.finish().await?;
    stats.output_batches = writer_result.output_batches;
    stats.writer_write_duration = writer_result.write_duration;
    stats.writer_encode_duration = writer_result.encode_duration;
    stats.writer_sink_duration = writer_result.sink_duration;
    stats.writer_close_duration = writer_result.close_duration;
    stats.copied_row_groups = writer_result.copied_row_groups;
    stats.copied_rows = writer_result.copied_rows;
    stats.copied_compressed_bytes = writer_result.copied_compressed_bytes;
    stats.row_group_copy_duration = writer_result.row_group_copy_duration;
    stats.writer_peak_buffered_bytes = writer_result.writer_peak_buffered_bytes;
    stats.ordered_pipeline_peak_buffered_bytes = ordered_memory_limiter
        .as_ref()
        .map(|limiter| limiter.peak_bytes())
        .unwrap_or_default();
    stats.read_decode_duration =
        Duration::from_nanos(worker_metrics.read_decode_nanos.load(Ordering::Relaxed));
    stats.source_prepare_duration =
        Duration::from_nanos(worker_metrics.source_prepare_nanos.load(Ordering::Relaxed));
    stats.execution_duration = execution_start.elapsed();
    Ok(stats)
}

fn select_fast_path_source_typed<K: OrderedKey>(
    sources: &[OrderedMergeSourceTyped<K>],
    source_order_metadata: &[SourceOrderMetadata],
) -> Option<usize> {
    let mut selected = None;
    for source_index in 0..sources.len() {
        if !row_group_fast_path_safe_typed::<K>(source_index, sources, source_order_metadata) {
            continue;
        }
        match selected {
            None => selected = Some(source_index),
            Some(current) => {
                if compare_source_heads_for_fast_path_typed::<K>(source_index, current, sources)
                    == Some(CmpOrdering::Less)
                {
                    selected = Some(source_index);
                }
            }
        }
    }
    selected
}

fn row_group_fast_path_safe_typed<K: OrderedKey>(
    candidate_index: usize,
    sources: &[OrderedMergeSourceTyped<K>],
    source_order_metadata: &[SourceOrderMetadata],
) -> bool {
    let Some(source) = sources.get(candidate_index) else {
        return false;
    };
    let range = if let Some(candidate) = source.current_candidate.as_ref() {
        &candidate.range
    } else if let Some(current_batch) = source.current_batch.as_ref() {
        let Some(range) = source_order_metadata
            .get(candidate_index)
            .and_then(|metadata| metadata.row_groups.get(current_batch.row_group_index))
        else {
            return false;
        };
        range
    } else {
        return false;
    };

    match range.null_kind {
        RowGroupNullKind::MixedOrUnknown => false,
        RowGroupNullKind::NoNulls => {
            let (Some(min), Some(max)) = (range.min.as_ref(), range.max.as_ref()) else {
                return false;
            };
            if compare_order_key_values(min, max) == CmpOrdering::Greater {
                return false;
            }
            for (competitor_index, competitor) in sources.iter().enumerate() {
                if competitor_index == candidate_index || !competitor.has_active_head() {
                    continue;
                }
                let Some(ordering) = compare_value_to_source_head_typed::<K>(max, competitor)
                else {
                    return false;
                };
                if ordering == CmpOrdering::Greater
                    || (ordering == CmpOrdering::Equal && candidate_index > competitor_index)
                {
                    return false;
                }
            }
            true
        }
        RowGroupNullKind::AllNulls => {
            for (competitor_index, competitor) in sources.iter().enumerate() {
                if competitor_index == candidate_index || !competitor.has_active_head() {
                    continue;
                }
                if competitor.may_have_non_null_remaining() || candidate_index > competitor_index {
                    return false;
                }
            }
            true
        }
    }
}

fn compare_value_to_source_head_typed<K: OrderedKey>(
    value: &OrderKeyValue,
    source: &OrderedMergeSourceTyped<K>,
) -> Option<CmpOrdering> {
    if let Some(batch) = source.current_batch.as_ref() {
        return Some(K::compare_to_value(&batch.order_column, source.current_row, value).reverse());
    }

    let candidate = source.current_candidate.as_ref()?;
    compare_value_to_candidate_head_typed::<K>(value, candidate)
}

fn compare_value_to_candidate_head_typed<K: OrderedKey>(
    value: &OrderKeyValue,
    candidate: &OrderedRowGroupCandidate,
) -> Option<CmpOrdering> {
    match candidate.range.null_kind {
        RowGroupNullKind::MixedOrUnknown => None,
        RowGroupNullKind::NoNulls => candidate
            .range
            .min
            .as_ref()
            .map(|min| compare_order_key_values(value, min)),
        RowGroupNullKind::AllNulls => Some(compare_order_key_values(value, &K::null_key_value())),
    }
}

fn compare_candidate_head_to_source_typed<K: OrderedKey>(
    candidate: &OrderedRowGroupCandidate,
    source: &OrderedMergeSourceTyped<K>,
) -> Option<CmpOrdering> {
    match candidate.range.null_kind {
        RowGroupNullKind::MixedOrUnknown => None,
        RowGroupNullKind::NoNulls => {
            let min = candidate.range.min.as_ref()?;
            compare_value_to_source_head_typed::<K>(min, source)
        }
        RowGroupNullKind::AllNulls => {
            if let Some(other_candidate) = source.current_candidate.as_ref() {
                return match other_candidate.range.null_kind {
                    RowGroupNullKind::MixedOrUnknown => None,
                    RowGroupNullKind::NoNulls => Some(CmpOrdering::Greater),
                    RowGroupNullKind::AllNulls => Some(CmpOrdering::Equal),
                };
            }
            if source.current_batch.is_some() {
                if source.has_non_null_remaining() {
                    Some(CmpOrdering::Greater)
                } else {
                    Some(CmpOrdering::Equal)
                }
            } else {
                None
            }
        }
    }
}

fn compare_source_heads_for_fast_path_typed<K: OrderedKey>(
    left_index: usize,
    right_index: usize,
    sources: &[OrderedMergeSourceTyped<K>],
) -> Option<CmpOrdering> {
    let left = sources.get(left_index)?;
    let right = sources.get(right_index)?;
    let ordering = if left.current_batch.is_some() && right.current_batch.is_some() {
        left.compare_head_to_source(right)
    } else if let Some(candidate) = left.current_candidate.as_ref() {
        compare_candidate_head_to_source_typed::<K>(candidate, right)?
    } else if let Some(candidate) = right.current_candidate.as_ref() {
        compare_candidate_head_to_source_typed::<K>(candidate, left)?.reverse()
    } else {
        return None;
    };
    if ordering == CmpOrdering::Equal {
        Some(left_index.cmp(&right_index))
    } else {
        Some(ordering)
    }
}

async fn drain_fast_path_source_typed<K: OrderedKey>(
    source_index: usize,
    sources: &mut [OrderedMergeSourceTyped<K>],
    source_order_metadata: &[SourceOrderMetadata],
    source_copy_plans: &[OrderedRowGroupCopyPlan],
    writer: &mut ParquetOutputWriter,
    output_pipeline: &mut OrderedOutputPipeline,
    execution_options: &ParquetMergeExecutionOptions,
    accumulator: &mut OrderedOutputAccumulator,
    stats: &mut ParquetMergeRunStats,
) -> Result<(u64, u64), String> {
    let mut drained_row_groups = 0_u64;
    let mut drained_batches = 0_u64;

    while row_group_fast_path_safe_typed::<K>(source_index, sources, source_order_metadata) {
        let current_row_group_index = sources[source_index]
            .current_row_group_index()
            .expect("fast path requires an active row group");
        drained_row_groups += 1;
        let mut counted_copy_candidate = false;

        if let Some(candidate) = sources[source_index].current_candidate.clone() {
            stats.copy_candidate_row_groups += 1;
            counted_copy_candidate = true;
            if writer.supports_row_group_copy() {
                flush_ordered_accumulator(accumulator, output_pipeline, writer, stats).await?;
                let copy = candidate.copy;
                stats.rows += copy.rows;
                sources[source_index]
                    .skip_current_candidate(&mut stats.input_batches)
                    .await?;
                output_pipeline
                    .submit_row_group_copy(copy, writer, stats)
                    .await?;
                continue;
            }

            sources[source_index]
                .decode_current_candidate(&mut stats.input_batches)
                .await?;
            if sources[source_index].current_batch.is_none() {
                break;
            }
        }

        if let Some(copy) = copy_request_for_current_row_group_typed(
            source_index,
            current_row_group_index,
            sources,
            source_copy_plans,
        ) {
            if !counted_copy_candidate {
                stats.copy_candidate_row_groups += 1;
            }
            if writer.supports_row_group_copy() {
                flush_ordered_accumulator(accumulator, output_pipeline, writer, stats).await?;
                let (batches, rows) = consume_current_row_group_without_writing_typed(
                    source_index,
                    current_row_group_index,
                    sources,
                    stats,
                )
                .await?;
                drained_batches += batches;
                stats.rows += rows;
                if rows != copy.rows {
                    return Err(format!(
                        "copied row group row count mismatch: decoded {rows}, metadata {}",
                        copy.rows
                    ));
                }
                output_pipeline
                    .submit_row_group_copy(copy, writer, stats)
                    .await?;
                continue;
            }
        }

        loop {
            let still_on_row_group = sources[source_index]
                .current_batch
                .as_ref()
                .map(|batch| batch.row_group_index == current_row_group_index)
                .unwrap_or(false);
            if !still_on_row_group {
                break;
            }

            let remaining_batch_rows = {
                let source = &sources[source_index];
                let batch = source
                    .current_batch
                    .as_ref()
                    .expect("row group drain requires an active batch");
                batch.batch.num_rows() - source.current_row
            };
            if sources[source_index].current_row == 0 {
                drained_batches += 1;
            }
            append_rows_from_source_typed::<K>(
                source_index,
                remaining_batch_rows,
                sources,
                writer,
                output_pipeline,
                execution_options,
                accumulator,
                stats,
            )
            .await?;
        }
    }

    Ok((drained_row_groups, drained_batches))
}

fn copy_request_for_current_row_group_typed<K: OrderedKey>(
    source_index: usize,
    row_group_index: usize,
    sources: &[OrderedMergeSourceTyped<K>],
    source_copy_plans: &[OrderedRowGroupCopyPlan],
) -> Option<RowGroupCopyRequest> {
    let source = sources.get(source_index)?;
    let current_batch = source.current_batch.as_ref()?;
    if source.current_row != 0
        || current_batch.row_group_index != row_group_index
        || current_batch.row_group_batch_index != 0
    {
        return None;
    }
    source_copy_plans
        .get(source_index)?
        .row_groups
        .get(row_group_index)?
        .clone()
}

async fn consume_current_row_group_without_writing_typed<K: OrderedKey>(
    source_index: usize,
    row_group_index: usize,
    sources: &mut [OrderedMergeSourceTyped<K>],
    stats: &mut ParquetMergeRunStats,
) -> Result<(u64, u64), String> {
    let mut batches = 0_u64;
    let mut rows = 0_u64;
    loop {
        let still_on_row_group = sources[source_index]
            .current_batch
            .as_ref()
            .map(|batch| batch.row_group_index == row_group_index)
            .unwrap_or(false);
        if !still_on_row_group {
            break;
        }
        let remaining_rows = {
            let source = &sources[source_index];
            let batch = source
                .current_batch
                .as_ref()
                .expect("row group copy requires an active batch");
            batch.batch.num_rows() - source.current_row
        };
        if sources[source_index].current_row == 0 {
            batches += 1;
        }
        rows += remaining_rows as u64;
        sources[source_index].current_row += remaining_rows;
        sources[source_index]
            .load_next(&mut stats.input_batches)
            .await?;
    }
    Ok((batches, rows))
}

async fn decode_pending_copy_candidates_typed<K: OrderedKey>(
    sources: &mut [OrderedMergeSourceTyped<K>],
    selector: &mut TypedSourceTournament<K>,
    stats: &mut ParquetMergeRunStats,
) -> Result<bool, String> {
    let mut decoded_any = false;
    for source_index in 0..sources.len() {
        if sources[source_index].current_candidate.is_none() {
            continue;
        }
        decoded_any = true;
        sources[source_index]
            .decode_current_candidate(&mut stats.input_batches)
            .await?;
        if sources[source_index].current_batch.is_some() {
            selector.set_active(source_index, true, sources);
        } else {
            selector.set_active(source_index, false, sources);
        }
    }
    Ok(decoded_any)
}

const DENSE_ORDERED_INITIAL_RUN_THRESHOLD: usize = 8;
const DENSE_ORDERED_MAX_PARTITIONS_PER_CORE: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DenseOrderedKeyValue {
    Int64(i64),
    String(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DenseOrderedBoundary {
    key: DenseOrderedKeyValue,
    source_index: usize,
    row: usize,
}

#[derive(Clone, Debug)]
enum DenseOrderedColumn {
    Int64(Int64Array),
    Utf8(StringArray),
    LargeUtf8(LargeStringArray),
}

impl DenseOrderedColumn {
    fn new(key_type: ParquetOrderKeyType, column: &ArrayRef) -> Option<Self> {
        match key_type {
            ParquetOrderKeyType::Int64 => column
                .as_any()
                .downcast_ref::<Int64Array>()
                .cloned()
                .map(Self::Int64),
            ParquetOrderKeyType::Utf8 => column
                .as_any()
                .downcast_ref::<StringArray>()
                .cloned()
                .map(Self::Utf8),
            ParquetOrderKeyType::LargeUtf8 => column
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .cloned()
                .map(Self::LargeUtf8),
            _ => None,
        }
    }

    fn key_at(&self, row: usize) -> DenseOrderedKeyValue {
        match self {
            Self::Int64(array) => DenseOrderedKeyValue::Int64(array.value(row)),
            Self::Utf8(array) => DenseOrderedKeyValue::String(array.value(row).to_string()),
            Self::LargeUtf8(array) => DenseOrderedKeyValue::String(array.value(row).to_string()),
        }
    }

    fn compare_row_to_key(&self, row: usize, key: &DenseOrderedKeyValue) -> CmpOrdering {
        match (self, key) {
            (Self::Int64(array), DenseOrderedKeyValue::Int64(key)) => array.value(row).cmp(key),
            (Self::Utf8(array), DenseOrderedKeyValue::String(key)) => array.value(row).cmp(key),
            (Self::LargeUtf8(array), DenseOrderedKeyValue::String(key)) => {
                array.value(row).cmp(key)
            }
            _ => CmpOrdering::Equal,
        }
    }

    fn compare_rows(
        &self,
        left_row: usize,
        other: &DenseOrderedColumn,
        right_row: usize,
    ) -> CmpOrdering {
        match (self, other) {
            (Self::Int64(left), Self::Int64(right)) => {
                left.value(left_row).cmp(&right.value(right_row))
            }
            (Self::Utf8(left), Self::Utf8(right)) => {
                left.value(left_row).cmp(right.value(right_row))
            }
            (Self::LargeUtf8(left), Self::LargeUtf8(right)) => {
                left.value(left_row).cmp(right.value(right_row))
            }
            _ => CmpOrdering::Equal,
        }
    }
}

#[derive(Clone, Debug)]
struct DenseOrderedSourceSnapshot {
    source_index: usize,
    batch: RecordBatch,
    order_column: DenseOrderedColumn,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct DenseOrderedPartitionJob {
    partition_index: usize,
    sources: Vec<DenseOrderedSourceSnapshot>,
    rows: usize,
}

#[derive(Debug)]
struct DenseOrderedPartitionOutput {
    partition_index: usize,
    batch: RecordBatch,
    rows: usize,
    mode: OrderedFlushMode,
    selection_duration: Duration,
    materialization_duration: Duration,
    comparisons: u64,
}

async fn try_dense_ordered_merge_window_typed<K: OrderedKey>(
    sources: &mut [OrderedMergeSourceTyped<K>],
    selector: &mut TypedSourceTournament<K>,
    writer: &mut ParquetOutputWriter,
    output_pipeline: &mut OrderedOutputPipeline,
    execution_options: &ParquetMergeExecutionOptions,
    accumulator: &mut OrderedOutputAccumulator,
    stats: &mut ParquetMergeRunStats,
    cpu_parallelism: usize,
    order_plan: &ParquetOrderPlan,
) -> Result<bool, String> {
    if !dense_ordered_key_supported(K::KEY_TYPE)
        || execution_options.parallelism == 1
        || execution_options.ordered_memory_budget_bytes.is_none()
        || cpu_parallelism <= 1
        || selector.active_count < 2
    {
        return Ok(false);
    }

    if sources
        .iter()
        .any(|source| source.current_candidate.is_some())
    {
        stats.dense_fallback_count += 1;
        return Ok(false);
    }

    let Some(entry) = selector.winner() else {
        return Ok(false);
    };
    let next_competitor = selector
        .next_competitor(entry.source_index, sources)
        .map(|entry| entry.source_index);
    let initial_run_len = {
        let source = &sources[entry.source_index];
        let competitor = next_competitor.map(|index| &sources[index]);
        source.contiguous_run_len(
            competitor,
            DENSE_ORDERED_INITIAL_RUN_THRESHOLD.saturating_add(1),
        )
    };
    if initial_run_len > DENSE_ORDERED_INITIAL_RUN_THRESHOLD {
        return Ok(false);
    }

    let mut snapshots = Vec::new();
    for (source_index, source) in sources.iter().enumerate() {
        if !selector.active.get(source_index).copied().unwrap_or(false) {
            continue;
        }
        let Some(batch) = source.current_batch.as_ref() else {
            stats.dense_fallback_count += 1;
            return Ok(false);
        };
        if source.current_row >= batch.non_null_prefix_len {
            stats.dense_fallback_count += 1;
            return Ok(false);
        }
        let Some(order_column) =
            DenseOrderedColumn::new(K::KEY_TYPE, batch.batch.column(order_plan.field_index))
        else {
            return Err(format!(
                "ordering_field `{}` is not {} at runtime",
                order_plan.field_name,
                K::RUNTIME_NAME
            ));
        };
        snapshots.push(DenseOrderedSourceSnapshot {
            source_index,
            batch: batch.batch.clone(),
            order_column,
            start: source.current_row,
            end: batch.non_null_prefix_len,
        });
    }

    let Some(safe_upper_bound) = dense_ordered_safe_upper_bound(&snapshots) else {
        return Ok(false);
    };
    for snapshot in &mut snapshots {
        snapshot.end = dense_ordered_bound(snapshot, &safe_upper_bound).min(snapshot.end);
    }
    snapshots.retain(|snapshot| snapshot.start < snapshot.end);
    if snapshots.len() < 2 {
        return Ok(false);
    }

    let total_rows = snapshots
        .iter()
        .map(|snapshot| snapshot.end.saturating_sub(snapshot.start))
        .sum::<usize>();
    if total_rows == 0 {
        return Ok(false);
    }

    let target_rows = dense_ordered_target_rows(K::KEY_TYPE, execution_options);
    let target_partitions = total_rows.div_ceil(target_rows).max(1);
    let max_partitions = cpu_parallelism
        .saturating_mul(DENSE_ORDERED_MAX_PARTITIONS_PER_CORE)
        .max(1);
    let partition_count = target_partitions.min(max_partitions).max(1);
    let partitions = dense_ordered_partition_jobs(&snapshots, partition_count);
    if partitions.is_empty() {
        stats.dense_fallback_count += 1;
        return Ok(false);
    }
    if partitions.len() > 1 {
        let largest_partition = partitions
            .iter()
            .map(|partition| partition.rows)
            .max()
            .unwrap_or_default();
        if largest_partition > total_rows.saturating_mul(9) / 10 {
            stats.dense_fallback_count += 1;
            return Ok(false);
        }
    }

    if !accumulator.is_empty() {
        flush_ordered_accumulator(accumulator, output_pipeline, writer, stats).await?;
    }

    let mut pending = FuturesUnordered::new();
    for partition in partitions {
        stats.dense_partition_jobs += 1;
        pending.push(tokio::task::spawn_blocking(move || {
            dense_ordered_materialize_partition(partition)
        }));
    }

    let mut completed = BTreeMap::new();
    let mut dense_selection_duration = Duration::default();
    let mut dense_materialization_duration = Duration::default();
    let mut dense_comparisons = 0_u64;
    while let Some(join_result) = pending.next().await {
        let output = match join_result {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => return Err(error),
            Err(error) => {
                return Err(format!(
                    "dense ordered materialization task failed to join: {error}"
                ));
            }
        };
        dense_selection_duration += output.selection_duration;
        dense_materialization_duration += output.materialization_duration;
        dense_comparisons += output.comparisons;
        completed.insert(output.partition_index, output);
    }

    add_ordered_selection_duration(stats, dense_selection_duration);
    stats.dense_selection_duration += dense_selection_duration;
    stats.dense_materialization_duration += dense_materialization_duration;
    stats.ordered_selector_comparisons += dense_comparisons + selector.take_comparisons();

    for (_, output) in completed {
        stats.rows += output.rows as u64;
        stats.dense_rows += output.rows as u64;
        output_pipeline
            .submit_materialized_batch(
                output.batch,
                output.mode,
                output.materialization_duration,
                writer,
                stats,
            )
            .await?;
    }

    for snapshot in snapshots {
        let source_index = snapshot.source_index;
        if sources[source_index].current_row == 0 {
            stats.fallback_batches += 1;
        }
        sources[source_index].current_row = snapshot.end;
        let exhausted_batch = sources[source_index]
            .current_batch
            .as_ref()
            .map(|batch| sources[source_index].current_row >= batch.batch.num_rows())
            .unwrap_or(false);
        if exhausted_batch {
            sources[source_index]
                .load_next(&mut stats.input_batches)
                .await?;
        }
        selector.set_active(
            source_index,
            sources[source_index].current_batch.is_some(),
            sources,
        );
    }
    stats.ordered_selector_comparisons += selector.take_comparisons();

    Ok(true)
}

fn dense_ordered_key_supported(key_type: ParquetOrderKeyType) -> bool {
    matches!(
        key_type,
        ParquetOrderKeyType::Int64 | ParquetOrderKeyType::Utf8 | ParquetOrderKeyType::LargeUtf8
    )
}

fn dense_ordered_target_rows(
    key_type: ParquetOrderKeyType,
    execution_options: &ParquetMergeExecutionOptions,
) -> usize {
    match key_type {
        ParquetOrderKeyType::Utf8 | ParquetOrderKeyType::LargeUtf8 => execution_options
            .output_row_group_rows
            .max(execution_options.output_batch_rows)
            .max(1),
        _ => execution_options.output_batch_rows.max(1),
    }
}

fn dense_ordered_partition_jobs(
    snapshots: &[DenseOrderedSourceSnapshot],
    partition_count: usize,
) -> Vec<DenseOrderedPartitionJob> {
    let total_rows = snapshots
        .iter()
        .map(|snapshot| snapshot.end.saturating_sub(snapshot.start))
        .sum::<usize>();
    if total_rows == 0 {
        return Vec::new();
    }

    let partition_count = partition_count.min(total_rows).max(1);
    let boundaries = dense_ordered_boundaries(snapshots, partition_count);
    let mut jobs = Vec::new();
    let mut lower = None;
    for partition_index in 0..=boundaries.len() {
        let upper = boundaries.get(partition_index);
        let mut partition_sources = Vec::new();
        let mut rows = 0_usize;
        for snapshot in snapshots {
            let start = lower
                .map(|boundary| dense_ordered_bound(snapshot, boundary))
                .unwrap_or(snapshot.start);
            let end = upper
                .map(|boundary| dense_ordered_bound(snapshot, boundary))
                .unwrap_or(snapshot.end);
            if start < end {
                rows += end - start;
                partition_sources.push(DenseOrderedSourceSnapshot {
                    source_index: snapshot.source_index,
                    batch: snapshot.batch.clone(),
                    order_column: snapshot.order_column.clone(),
                    start,
                    end,
                });
            }
        }
        if rows > 0 {
            jobs.push(DenseOrderedPartitionJob {
                partition_index: jobs.len(),
                sources: partition_sources,
                rows,
            });
        }
        lower = upper;
    }
    jobs
}

fn dense_ordered_safe_upper_bound(
    snapshots: &[DenseOrderedSourceSnapshot],
) -> Option<DenseOrderedBoundary> {
    snapshots
        .iter()
        .filter(|snapshot| snapshot.start < snapshot.end)
        .map(|snapshot| DenseOrderedBoundary {
            key: snapshot.order_column.key_at(snapshot.end - 1),
            source_index: snapshot.source_index,
            row: snapshot.end,
        })
        .min()
}

fn dense_ordered_boundaries(
    snapshots: &[DenseOrderedSourceSnapshot],
    partition_count: usize,
) -> Vec<DenseOrderedBoundary> {
    if partition_count <= 1 {
        return Vec::new();
    }

    let sample_target = partition_count.saturating_mul(8).max(1);
    let mut samples = Vec::new();
    for snapshot in snapshots {
        let len = snapshot.end.saturating_sub(snapshot.start);
        if len == 0 {
            continue;
        }
        let stride = len.div_ceil(sample_target).max(1);
        let mut row = snapshot.start;
        while row < snapshot.end {
            samples.push(DenseOrderedBoundary {
                key: snapshot.order_column.key_at(row),
                source_index: snapshot.source_index,
                row,
            });
            row = row.saturating_add(stride);
        }
        samples.push(DenseOrderedBoundary {
            key: snapshot.order_column.key_at(snapshot.end - 1),
            source_index: snapshot.source_index,
            row: snapshot.end - 1,
        });
    }
    samples.sort_unstable();
    samples.dedup();
    if samples.is_empty() {
        return Vec::new();
    }

    let mut boundaries = Vec::new();
    for partition in 1..partition_count {
        let sample_index = samples.len().saturating_mul(partition) / partition_count;
        if let Some(boundary) = samples.get(sample_index.min(samples.len() - 1)).cloned() {
            if boundaries.last() != Some(&boundary) {
                boundaries.push(boundary);
            }
        }
    }
    boundaries
}

fn dense_ordered_bound(
    snapshot: &DenseOrderedSourceSnapshot,
    boundary: &DenseOrderedBoundary,
) -> usize {
    let mut low = snapshot.start;
    let mut high = snapshot.end;
    while low < high {
        let mid = low + ((high - low) / 2);
        if dense_ordered_row_before_boundary(snapshot, mid, boundary) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

fn dense_ordered_row_before_boundary(
    snapshot: &DenseOrderedSourceSnapshot,
    row: usize,
    boundary: &DenseOrderedBoundary,
) -> bool {
    match snapshot.order_column.compare_row_to_key(row, &boundary.key) {
        CmpOrdering::Less => true,
        CmpOrdering::Greater => false,
        CmpOrdering::Equal => match snapshot.source_index.cmp(&boundary.source_index) {
            CmpOrdering::Less => true,
            CmpOrdering::Greater => false,
            CmpOrdering::Equal => row < boundary.row,
        },
    }
}

fn dense_ordered_materialize_partition(
    partition: DenseOrderedPartitionJob,
) -> Result<DenseOrderedPartitionOutput, String> {
    let selection_start = Instant::now();
    if partition.sources.len() == 1 {
        let source = partition
            .sources
            .into_iter()
            .next()
            .expect("single dense partition source exists");
        let batch = source.batch.slice(source.start, source.end - source.start);
        return Ok(DenseOrderedPartitionOutput {
            partition_index: partition.partition_index,
            batch,
            rows: partition.rows,
            mode: OrderedFlushMode::Direct,
            selection_duration: selection_start.elapsed(),
            materialization_duration: Duration::default(),
            comparisons: 0,
        });
    }

    let mut positions = partition
        .sources
        .iter()
        .map(|source| source.start)
        .collect::<Vec<_>>();
    let mut indices = Vec::with_capacity(partition.rows);
    let mut comparisons = 0_u64;
    while indices.len() < partition.rows {
        let mut winner = None;
        for local_index in 0..partition.sources.len() {
            if positions[local_index] >= partition.sources[local_index].end {
                continue;
            }
            winner = Some(match winner {
                None => local_index,
                Some(current) => {
                    comparisons += 1;
                    if dense_ordered_source_row_less(
                        &partition.sources[local_index],
                        positions[local_index],
                        &partition.sources[current],
                        positions[current],
                    ) {
                        local_index
                    } else {
                        current
                    }
                }
            });
        }
        let Some(winner) = winner else {
            return Err("dense ordered partition exhausted before expected row count".to_string());
        };
        indices.push((winner, positions[winner]));
        positions[winner] += 1;
    }
    let selection_duration = selection_start.elapsed();

    let materialization_start = Instant::now();
    let batch_refs = partition
        .sources
        .iter()
        .map(|source| &source.batch)
        .collect::<Vec<_>>();
    let batch = interleave_record_batch(&batch_refs, &indices)
        .map_err(|error| format!("failed to materialize dense ordered partition: {error}"))?;
    Ok(DenseOrderedPartitionOutput {
        partition_index: partition.partition_index,
        batch,
        rows: partition.rows,
        mode: OrderedFlushMode::Interleave,
        selection_duration,
        materialization_duration: materialization_start.elapsed(),
        comparisons,
    })
}

fn dense_ordered_source_row_less(
    left: &DenseOrderedSourceSnapshot,
    left_row: usize,
    right: &DenseOrderedSourceSnapshot,
    right_row: usize,
) -> bool {
    match left
        .order_column
        .compare_rows(left_row, &right.order_column, right_row)
    {
        CmpOrdering::Less => true,
        CmpOrdering::Greater => false,
        CmpOrdering::Equal => match left.source_index.cmp(&right.source_index) {
            CmpOrdering::Less => true,
            CmpOrdering::Greater => false,
            CmpOrdering::Equal => left_row < right_row,
        },
    }
}

fn add_ordered_selection_duration(stats: &mut ParquetMergeRunStats, duration: Duration) {
    stats.ordered_output_selection_duration += duration;
    stats.ordered_output_assembly_duration += duration;
}

fn add_ordered_materialization_duration(
    stats: &mut ParquetMergeRunStats,
    duration: Duration,
    mode: OrderedFlushMode,
) {
    stats.ordered_output_materialization_duration += duration;
    stats.ordered_output_assembly_duration += duration;
    stats.accumulator_flushes += 1;
    match mode {
        OrderedFlushMode::Direct => {}
        OrderedFlushMode::Concat => stats.accumulator_concat_flushes += 1,
        OrderedFlushMode::Interleave => stats.accumulator_interleave_flushes += 1,
    }
}

async fn flush_ordered_accumulator(
    accumulator: &mut OrderedOutputAccumulator,
    output_pipeline: &mut OrderedOutputPipeline,
    writer: &mut ParquetOutputWriter,
    stats: &mut ParquetMergeRunStats,
) -> Result<(), String> {
    let flush_start = Instant::now();
    if let Some(segment) = accumulator.flush_segment()? {
        let direct_materialization_duration = flush_start.elapsed();
        output_pipeline
            .submit_flush_segment(segment, writer, stats, direct_materialization_duration)
            .await?;
    }
    Ok(())
}

async fn append_rows_from_source_typed<K: OrderedKey>(
    source_index: usize,
    row_count: usize,
    sources: &mut [OrderedMergeSourceTyped<K>],
    writer: &mut ParquetOutputWriter,
    output_pipeline: &mut OrderedOutputPipeline,
    execution_options: &ParquetMergeExecutionOptions,
    accumulator: &mut OrderedOutputAccumulator,
    stats: &mut ParquetMergeRunStats,
) -> Result<(), String> {
    let mut remaining = row_count;
    while remaining > 0 {
        let (current_row, batch_rows, can_write_whole_batch_directly) = {
            let source = &sources[source_index];
            let batch = source
                .current_batch
                .as_ref()
                .expect("appending rows requires an active batch");
            let batch_rows = batch.batch.num_rows() - source.current_row;
            let can_write_whole_batch_directly = source.current_row == 0
                && remaining >= batch_rows
                && batch_rows <= execution_options.output_batch_rows;
            (
                source.current_row,
                batch_rows,
                can_write_whole_batch_directly,
            )
        };

        if can_write_whole_batch_directly && !accumulator.is_empty() {
            flush_ordered_accumulator(accumulator, output_pipeline, writer, stats).await?;
            continue;
        }

        let whole_batch_direct = accumulator.is_empty() && can_write_whole_batch_directly;
        if whole_batch_direct {
            let direct_batch = sources[source_index]
                .current_batch
                .as_ref()
                .expect("direct batch write requires an active batch");
            let direct_batch = direct_batch.batch.clone();
            output_pipeline
                .submit_direct_batch(direct_batch, writer, stats)
                .await?;
            stats.direct_batch_writes += 1;
            stats.rows += batch_rows as u64;
            remaining -= batch_rows;
            sources[source_index].current_row += batch_rows;
        } else {
            let take = batch_rows.min(remaining).min(
                execution_options
                    .output_batch_rows
                    .saturating_sub(accumulator.rows())
                    .max(1),
            );
            let batch = sources[source_index]
                .current_batch
                .as_ref()
                .expect("pending output append requires an active batch");
            let assembly_start = Instant::now();
            accumulator.append_range(&batch.batch, current_row, take)?;
            add_ordered_selection_duration(stats, assembly_start.elapsed());
            stats.rows += take as u64;
            remaining -= take;
            sources[source_index].current_row += take;

            if accumulator.rows() >= execution_options.output_batch_rows {
                flush_ordered_accumulator(accumulator, output_pipeline, writer, stats).await?;
            }
        }

        let exhausted_batch = sources[source_index]
            .current_batch
            .as_ref()
            .map(|batch| sources[source_index].current_row >= batch.batch.num_rows())
            .unwrap_or(false);
        if exhausted_batch {
            sources[source_index]
                .load_next(&mut stats.input_batches)
                .await?;
        }
    }

    Ok(())
}

async fn parquet_merge_source_worker_typed<K: OrderedKey>(
    source_index: usize,
    input_path: PathBuf,
    output_schema: SchemaRef,
    adapter: Arc<CompiledSourceAdapter>,
    order_plan: Arc<ParquetOrderPlan>,
    execution_options: ParquetMergeExecutionOptions,
    semaphore: Arc<Semaphore>,
    worker_metrics: Arc<ParquetWorkerMetrics>,
    source_order_metadata: SourceOrderMetadata,
    source_copy_plan: OrderedRowGroupCopyPlan,
    mut command_rx: mpsc::Receiver<OrderedSourceCommand>,
    tx: mpsc::Sender<Result<OrderedSourceMessage<K>, String>>,
) -> Result<(), String> {
    let result: Result<(), String> = async {
        let mut file = File::open(&input_path)
            .await
            .map_err(|error| format!("failed to open `{}`: {error}", input_path.display()))?;
        let metadata = ArrowReaderMetadata::load_async(&mut file, Default::default())
            .await
            .map_err(|error| format!("failed to inspect `{}`: {error}", input_path.display()))?;
        let row_group_count = metadata.metadata().row_groups().len();
        let mut scratch = ExecutionScratch::default();
        let mut last_key = None;

        for row_group_index in 0..row_group_count {
            if let Some(copy) = source_copy_plan
                .row_groups
                .get(row_group_index)
                .and_then(Clone::clone)
            {
                let range = source_order_metadata
                    .row_groups
                    .get(row_group_index)
                    .cloned()
                    .ok_or_else(|| {
                        format!(
                            "missing order metadata for `{}` row group {}",
                            input_path.display(),
                            row_group_index
                        )
                    })?;
                let candidate = OrderedRowGroupCandidate {
                    row_group_index,
                    range,
                    copy,
                };
                if tx
                    .send(Ok(OrderedSourceMessage::CopyCandidate(candidate.clone())))
                    .await
                    .is_err()
                {
                    return Ok(());
                }
                match command_rx.recv().await {
                    Some(OrderedSourceCommand::Skip) => {
                        validate_skipped_ordered_row_group_typed::<K>(
                            &candidate.range,
                            &mut last_key,
                            &input_path,
                            &order_plan.field_name,
                            source_index,
                        )?;
                        continue;
                    }
                    Some(OrderedSourceCommand::Decode) => {}
                    None => return Ok(()),
                }
            }

            let mut row_group_batch_index = 0_usize;
            let row_group_file = file.try_clone().await.map_err(|error| {
                format!(
                    "failed cloning parquet handle for `{}` row group {}: {error}",
                    input_path.display(),
                    row_group_index
                )
            })?;
            let builder = ParquetRecordBatchStreamBuilder::new_with_metadata(
                row_group_file,
                metadata.clone(),
            );
            let mut stream = builder
                .with_row_groups(vec![row_group_index])
                .with_batch_size(execution_options.read_batch_size)
                .build()
                .map_err(|error| {
                    format!(
                        "failed to build parquet stream for `{}` row group {}: {error}",
                        input_path.display(),
                        row_group_index
                    )
                })?;

            loop {
                let batch_result = {
                    let _permit =
                        semaphore.clone().acquire_owned().await.map_err(|_| {
                            "parquet merge parallelism semaphore closed".to_string()
                        })?;
                    let decode_start = Instant::now();
                    let batch_result = stream.next().await;
                    worker_metrics
                        .read_decode_nanos
                        .fetch_add(decode_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    batch_result
                };
                let Some(batch_result) = batch_result else {
                    break;
                };
                let batch = batch_result.map_err(|error| {
                    format!(
                        "failed reading parquet batch from `{}` row group {}: {error}",
                        input_path.display(),
                        row_group_index
                    )
                })?;
                let prepared = {
                    let _permit =
                        semaphore.clone().acquire_owned().await.map_err(|_| {
                            "parquet merge parallelism semaphore closed".to_string()
                        })?;
                    let prepare_start = Instant::now();
                    let adapted = adapter
                        .adapt_batch(&batch, &output_schema, &mut scratch)
                        .map_err(|error| {
                            format!(
                                "failed adapting parquet batch from `{}` row group {}: {error}",
                                input_path.display(),
                                row_group_index
                            )
                        })?;
                    let prepared = PreparedOrderBatchTyped::<K>::new(
                        adapted,
                        order_plan.as_ref(),
                        row_group_index,
                        row_group_batch_index,
                    )?;
                    row_group_batch_index += 1;
                    validate_prepared_ordered_batch_typed::<K>(
                        &prepared,
                        &mut last_key,
                        &input_path,
                        &order_plan.field_name,
                        source_index,
                    )?;
                    worker_metrics
                        .source_prepare_nanos
                        .fetch_add(prepare_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    prepared
                };
                if tx
                    .send(Ok(OrderedSourceMessage::Batch(prepared)))
                    .await
                    .is_err()
                {
                    return Ok(());
                }
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

fn validate_prepared_ordered_batch_typed<K: OrderedKey>(
    batch: &PreparedOrderBatchTyped<K>,
    last_key: &mut Option<OrderKeyValue>,
    input_path: &Path,
    field_name: &str,
    source_index: usize,
) -> Result<(), String> {
    if batch.batch.num_rows() == 0 {
        return Ok(());
    }

    if let Some(previous_key) = last_key.as_ref() {
        if K::compare_to_value(&batch.order_column, 0, previous_key) == CmpOrdering::Less {
            return Err(format!(
                "input parquet file `{}` is not sorted ascending by `{}` for source {}",
                input_path.display(),
                field_name,
                source_index
            ));
        }
    }

    for row in 1..batch.batch.num_rows() {
        if K::compare_values(&batch.order_column, row - 1, row) == CmpOrdering::Greater {
            return Err(format!(
                "input parquet file `{}` is not sorted ascending by `{}` near row {}",
                input_path.display(),
                field_name,
                row
            ));
        }
    }

    *last_key = Some(K::key_at(&batch.order_column, batch.batch.num_rows() - 1));
    Ok(())
}

fn validate_skipped_ordered_row_group_typed<K: OrderedKey>(
    range: &RowGroupOrderRange,
    last_key: &mut Option<OrderKeyValue>,
    input_path: &Path,
    field_name: &str,
    source_index: usize,
) -> Result<(), String> {
    let (first_key, last_row_key) = match range.null_kind {
        RowGroupNullKind::NoNulls => {
            let (Some(min), Some(max)) = (range.min.as_ref(), range.max.as_ref()) else {
                return Err(format!(
                    "copyable row group {} in `{}` is missing exact `{}` min/max statistics",
                    range.row_group_index,
                    input_path.display(),
                    field_name
                ));
            };
            if compare_order_key_values(min, max) == CmpOrdering::Greater {
                return Err(format!(
                    "copyable row group {} in `{}` has inverted `{}` statistics",
                    range.row_group_index,
                    input_path.display(),
                    field_name
                ));
            }
            (min.clone(), max.clone())
        }
        RowGroupNullKind::AllNulls => {
            let null_key = K::null_key_value();
            (null_key.clone(), null_key)
        }
        RowGroupNullKind::MixedOrUnknown => {
            return Err(format!(
                "copyable row group {} in `{}` cannot be skipped without exact null-order metadata",
                range.row_group_index,
                input_path.display()
            ));
        }
    };

    if let Some(previous_key) = last_key.as_ref() {
        if compare_order_key_values(previous_key, &first_key) == CmpOrdering::Greater {
            return Err(format!(
                "input parquet file `{}` is not sorted ascending by `{}` for source {}",
                input_path.display(),
                field_name,
                source_index
            ));
        }
    }
    *last_key = Some(last_row_key);
    Ok(())
}

#[cfg(test)]
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
        stats_fast_path_duration: Duration::default(),
        fast_path_row_groups: 0,
        fast_path_batches: 0,
        fallback_batches: 0,
        read_decode_duration: Duration::default(),
        source_prepare_duration: Duration::default(),
        ordered_output_assembly_duration: Duration::default(),
        ordered_output_selection_duration: Duration::default(),
        ordered_output_materialization_duration: Duration::default(),
        ordered_output_materialization_wait_duration: Duration::default(),
        ordered_pipeline_peak_buffered_bytes: 0,
        writer_peak_buffered_bytes: 0,
        ordered_selector_comparisons: 0,
        dense_partition_jobs: 0,
        dense_rows: 0,
        dense_selection_duration: Duration::default(),
        dense_materialization_duration: Duration::default(),
        dense_fallback_count: 0,
        writer_write_duration: Duration::default(),
        writer_encode_duration: Duration::default(),
        writer_sink_duration: Duration::default(),
        writer_close_duration: Duration::default(),
        direct_batch_writes: 0,
        accumulator_flushes: 0,
        accumulator_concat_flushes: 0,
        accumulator_interleave_flushes: 0,
        copy_candidate_row_groups: 0,
        copied_row_groups: 0,
        copied_rows: 0,
        copied_compressed_bytes: 0,
        row_group_copy_duration: Duration::default(),
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

fn compare_int64_arrays(
    left: &Int64Array,
    left_row: usize,
    right: &Int64Array,
    right_row: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(left.is_null(left_row), right.is_null(right_row)) {
        return ordering;
    }
    left.value(left_row).cmp(&right.value(right_row))
}

fn compare_uint64_values(array: &UInt64Array, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(&array.value(right))
}

fn compare_uint64_arrays(
    left: &UInt64Array,
    left_row: usize,
    right: &UInt64Array,
    right_row: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(left.is_null(left_row), right.is_null(right_row)) {
        return ordering;
    }
    left.value(left_row).cmp(&right.value(right_row))
}

fn compare_float64_values(array: &Float64Array, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).total_cmp(&array.value(right))
}

fn compare_float64_arrays(
    left: &Float64Array,
    left_row: usize,
    right: &Float64Array,
    right_row: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(left.is_null(left_row), right.is_null(right_row)) {
        return ordering;
    }
    left.value(left_row).total_cmp(&right.value(right_row))
}

fn compare_string_values(array: &StringArray, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(array.value(right))
}

fn compare_string_arrays(
    left: &StringArray,
    left_row: usize,
    right: &StringArray,
    right_row: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(left.is_null(left_row), right.is_null(right_row)) {
        return ordering;
    }
    left.value(left_row).cmp(right.value(right_row))
}

fn compare_large_string_values(array: &LargeStringArray, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(array.value(right))
}

fn compare_large_string_arrays(
    left: &LargeStringArray,
    left_row: usize,
    right: &LargeStringArray,
    right_row: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(left.is_null(left_row), right.is_null(right_row)) {
        return ordering;
    }
    left.value(left_row).cmp(right.value(right_row))
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

fn compare_date32_arrays(
    left: &Date32Array,
    left_row: usize,
    right: &Date32Array,
    right_row: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(left.is_null(left_row), right.is_null(right_row)) {
        return ordering;
    }
    left.value(left_row).cmp(&right.value(right_row))
}

fn compare_date64_values(array: &Date64Array, left: usize, right: usize) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(array.is_null(left), array.is_null(right)) {
        return ordering;
    }
    array.value(left).cmp(&array.value(right))
}

fn compare_date64_arrays(
    left: &Date64Array,
    left_row: usize,
    right: &Date64Array,
    right_row: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(left.is_null(left_row), right.is_null(right_row)) {
        return ordering;
    }
    left.value(left_row).cmp(&right.value(right_row))
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

fn compare_timestamp_second_arrays(
    left: &TimestampSecondArray,
    left_row: usize,
    right: &TimestampSecondArray,
    right_row: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(left.is_null(left_row), right.is_null(right_row)) {
        return ordering;
    }
    left.value(left_row).cmp(&right.value(right_row))
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

fn compare_timestamp_millisecond_arrays(
    left: &TimestampMillisecondArray,
    left_row: usize,
    right: &TimestampMillisecondArray,
    right_row: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(left.is_null(left_row), right.is_null(right_row)) {
        return ordering;
    }
    left.value(left_row).cmp(&right.value(right_row))
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

fn compare_timestamp_microsecond_arrays(
    left: &TimestampMicrosecondArray,
    left_row: usize,
    right: &TimestampMicrosecondArray,
    right_row: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(left.is_null(left_row), right.is_null(right_row)) {
        return ordering;
    }
    left.value(left_row).cmp(&right.value(right_row))
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

fn compare_timestamp_nanosecond_arrays(
    left: &TimestampNanosecondArray,
    left_row: usize,
    right: &TimestampNanosecondArray,
    right_row: usize,
) -> CmpOrdering {
    if let Some(ordering) = compare_null_flags(left.is_null(left_row), right.is_null(right_row)) {
        return ordering;
    }
    left.value(left_row).cmp(&right.value(right_row))
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

    if matches!(source_type, DataType::Int32) && matches!(target_type, DataType::Float64) {
        return Ok(CompiledTypeAdapter::Int32ToFloat64);
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
mod tests;
