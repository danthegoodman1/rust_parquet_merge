use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fs::File as StdFile;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
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
    StringBuilder, StructBuilder, UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
};
use arrow_array::{Array, ArrayRef, LargeListArray, ListArray, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use futures_util::StreamExt;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_writer::AsyncArrowWriter;
use serde_json::{Deserializer, Map, Number, Value};
use tokio::fs::File;
use tokio::io::BufWriter;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionOptions {
    pub envelope_fields: Vec<String>,
    pub payload_column: String,
    pub widening_options: WideningOptions,
    pub batch_rows: usize,
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            envelope_fields: Vec::new(),
            payload_column: "payload".to_string(),
            widening_options: WideningOptions::default(),
            batch_rows: 8_192,
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
    })
}

pub fn discover_ndjson_schema_from_paths(
    input_paths: &[PathBuf],
    options: &CompactionOptions,
) -> Result<SchemaRef, Box<dyn Error>> {
    validate_compaction_options(options).map_err(io_error)?;
    if input_paths.is_empty() {
        return Err(io_error("at least one NDJSON input path is required").into());
    }

    let envelope_names: BTreeSet<&str> = options
        .envelope_fields
        .iter()
        .map(|field| field.as_str())
        .collect();
    let mut envelope_types: Vec<Option<DataType>> = vec![None; options.envelope_fields.len()];
    let mut envelope_nullable = vec![false; options.envelope_fields.len()];
    let mut payload_type: Option<DataType> = None;

    for input_path in input_paths {
        let reader = BufReader::new(StdFile::open(input_path)?);
        let stream = Deserializer::from_reader(reader).into_iter::<Value>();

        for value in stream {
            let value = value?;
            let object = value.as_object().ok_or_else(|| {
                io_error(format!(
                    "NDJSON record in `{}` must be an object",
                    input_path.display()
                ))
            })?;

            for (index, field_name) in options.envelope_fields.iter().enumerate() {
                match object.get(field_name) {
                    Some(Value::Null) | None => envelope_nullable[index] = true,
                    Some(value) => {
                        let inferred = infer_json_data_type(value, &options.widening_options)?;
                        if let Some(existing) = &envelope_types[index] {
                            if existing != &inferred {
                                return Err(io_error(format!(
                                    "envelope column `{field_name}` changed types across NDJSON records: left={existing:?}, right={inferred:?}"
                                ))
                                .into());
                            }
                        } else {
                            envelope_types[index] = Some(inferred);
                        }
                    }
                }
            }

            let inferred_payload =
                infer_payload_type(object, &envelope_names, &options.widening_options)?;
            payload_type = match payload_type {
                Some(current) => Some(
                    widen_data_type(&current, &inferred_payload, &options.widening_options)
                        .map_err(|error| io_error(prefix_payload_conflicts(&error)))?,
                ),
                None => Some(inferred_payload),
            };
        }
    }

    let mut fields = Vec::with_capacity(options.envelope_fields.len() + 1);
    for (index, field_name) in options.envelope_fields.iter().enumerate() {
        let data_type = envelope_types[index].clone().unwrap_or(DataType::Null);
        fields.push(Arc::new(Field::new(
            field_name,
            data_type,
            envelope_nullable[index] || matches!(envelope_types[index], None),
        )));
    }

    fields.push(Arc::new(Field::new(
        &options.payload_column,
        payload_type.unwrap_or(DataType::Struct(Vec::<FieldRef>::new().into())),
        true,
    )));

    Ok(Arc::new(Schema::new(fields)))
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
    let output_schema = discover_ndjson_schema_from_paths(input_paths, options)?;
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
    let mut writer = AsyncArrowWriter::try_new(output_file, output_schema.clone(), None)?;
    let mut batch_builder = NdjsonBatchBuilder::new(
        output_schema.clone(),
        &options.payload_column,
        options.batch_rows,
    );
    let mut rows = 0_u64;

    for input_path in input_paths {
        let reader = BufReader::new(StdFile::open(input_path)?);
        let stream = Deserializer::from_reader(reader).into_iter::<Value>();

        for value in stream {
            let value = value?;
            let object = value.as_object().ok_or_else(|| {
                io_error(format!(
                    "NDJSON record in `{}` must be an object",
                    input_path.display()
                ))
            })?;

            batch_builder.append_record(object).map_err(io_error)?;
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
    })
}

struct NdjsonBatchBuilder {
    fields: Fields,
    payload_column: String,
    batch_rows: usize,
    builder: StructBuilder,
}

impl NdjsonBatchBuilder {
    fn new(schema: SchemaRef, payload_column: &str, batch_rows: usize) -> Self {
        let fields = schema.fields().clone();
        Self {
            fields: fields.clone(),
            payload_column: payload_column.to_string(),
            batch_rows,
            builder: StructBuilder::from_fields(fields, batch_rows),
        }
    }

    fn is_empty(&self) -> bool {
        self.builder.len() == 0
    }

    fn is_full(&self) -> bool {
        self.builder.len() >= self.batch_rows
    }

    fn append_record(&mut self, object: &Map<String, Value>) -> Result<(), String> {
        let builders = self.builder.field_builders_mut();
        for (index, field) in self.fields.iter().enumerate() {
            if field.name() == &self.payload_column {
                let struct_builder = builders[index]
                    .as_any_mut()
                    .downcast_mut::<StructBuilder>()
                    .ok_or_else(|| {
                        format!(
                            "payload column `{}` must be Struct, found {:?}",
                            field.name(),
                            field.data_type()
                        )
                    })?;
                let DataType::Struct(payload_fields) = field.data_type() else {
                    return Err(format!(
                        "payload column `{}` must be Struct, found {:?}",
                        field.name(),
                        field.data_type()
                    ));
                };
                append_struct_from_object(struct_builder, payload_fields, Some(object))?;
            } else {
                append_json_value(
                    builders[index].as_mut(),
                    field.data_type(),
                    object.get(field.name()),
                )?;
            }
        }
        self.builder.append(true);
        Ok(())
    }

    fn finish_batch(&mut self) -> RecordBatch {
        RecordBatch::from(self.builder.finish())
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

fn infer_payload_type(
    object: &Map<String, Value>,
    envelope_names: &BTreeSet<&str>,
    options: &WideningOptions,
) -> Result<DataType, String> {
    let mut payload_names: Vec<&str> = object
        .keys()
        .map(String::as_str)
        .filter(|name| !envelope_names.contains(name))
        .collect();
    payload_names.sort_unstable();

    let mut fields = Vec::with_capacity(payload_names.len());
    for name in payload_names {
        let value = object
            .get(name)
            .ok_or_else(|| format!("payload field `{name}` missing during schema inference"))?;
        fields.push(Arc::new(Field::new(
            name,
            infer_json_data_type(value, options)?,
            matches!(value, Value::Null),
        )));
    }

    Ok(DataType::Struct(fields.into()))
}

fn infer_json_data_type(value: &Value, options: &WideningOptions) -> Result<DataType, String> {
    match value {
        Value::Null => Ok(DataType::Null),
        Value::Bool(_) => Ok(DataType::Boolean),
        Value::Number(number) => infer_number_type(number),
        Value::String(_) => Ok(DataType::Utf8),
        Value::Array(values) => {
            let mut element_type: Option<DataType> = None;
            for value in values {
                let inferred = infer_json_data_type(value, options)?;
                element_type = match element_type {
                    Some(current) => Some(widen_data_type(&current, &inferred, options)?),
                    None => Some(inferred),
                };
            }

            Ok(DataType::List(Arc::new(Field::new(
                "item",
                element_type.unwrap_or(DataType::Null),
                true,
            ))))
        }
        Value::Object(object) => {
            let mut names: Vec<&str> = object.keys().map(String::as_str).collect();
            names.sort_unstable();

            let mut fields = Vec::with_capacity(names.len());
            for name in names {
                let value = object.get(name).ok_or_else(|| {
                    format!("object field `{name}` missing during schema inference")
                })?;
                fields.push(Arc::new(Field::new(
                    name,
                    infer_json_data_type(value, options)?,
                    matches!(value, Value::Null),
                )));
            }
            Ok(DataType::Struct(fields.into()))
        }
    }
}

fn infer_number_type(number: &Number) -> Result<DataType, String> {
    if number.is_i64() {
        Ok(DataType::Int64)
    } else if number.is_u64() {
        Ok(DataType::UInt64)
    } else if number.is_f64() {
        Ok(DataType::Float64)
    } else {
        Err(format!("unsupported JSON number `{number}`"))
    }
}

fn append_struct_from_object(
    builder: &mut StructBuilder,
    target_fields: &Fields,
    object: Option<&Map<String, Value>>,
) -> Result<(), String> {
    let child_builders = builder.field_builders_mut();
    for (index, field) in target_fields.iter().enumerate() {
        let value = object.and_then(|object| object.get(field.name()));
        append_json_value(child_builders[index].as_mut(), field.data_type(), value)?;
    }
    builder.append(object.is_some());
    Ok(())
}

fn append_json_value(
    builder: &mut dyn ArrayBuilder,
    target_type: &DataType,
    value: Option<&Value>,
) -> Result<(), String> {
    match target_type {
        DataType::Null => {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<NullBuilder>()
                .ok_or_else(|| "failed to downcast NullBuilder".to_string())?;
            match value {
                Some(Value::Null) | None => builder.append_null(),
                Some(other) => {
                    return Err(format!(
                        "expected null value for Null field, found {}",
                        json_kind(other)
                    ));
                }
            }
            Ok(())
        }
        DataType::Boolean => {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<BooleanBuilder>()
                .ok_or_else(|| "failed to downcast BooleanBuilder".to_string())?;
            match value {
                Some(Value::Bool(value)) => builder.append_value(*value),
                Some(Value::Null) | None => builder.append_null(),
                Some(other) => {
                    return Err(format!(
                        "expected boolean for Boolean field, found {}",
                        json_kind(other)
                    ));
                }
            }
            Ok(())
        }
        DataType::Int8 => append_signed_json(builder, value, "Int8", |builder, value| {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<Int8Builder>()
                .ok_or_else(|| "failed to downcast Int8Builder".to_string())?;
            builder.append_value(value);
            Ok(())
        }),
        DataType::Int16 => append_signed_json(builder, value, "Int16", |builder, value| {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<Int16Builder>()
                .ok_or_else(|| "failed to downcast Int16Builder".to_string())?;
            builder.append_value(value);
            Ok(())
        }),
        DataType::Int32 => append_signed_json(builder, value, "Int32", |builder, value| {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<Int32Builder>()
                .ok_or_else(|| "failed to downcast Int32Builder".to_string())?;
            builder.append_value(value);
            Ok(())
        }),
        DataType::Int64 => append_signed_json(builder, value, "Int64", |builder, value| {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<Int64Builder>()
                .ok_or_else(|| "failed to downcast Int64Builder".to_string())?;
            builder.append_value(value);
            Ok(())
        }),
        DataType::UInt8 => append_unsigned_json(builder, value, "UInt8", |builder, value| {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<UInt8Builder>()
                .ok_or_else(|| "failed to downcast UInt8Builder".to_string())?;
            builder.append_value(value);
            Ok(())
        }),
        DataType::UInt16 => append_unsigned_json(builder, value, "UInt16", |builder, value| {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<UInt16Builder>()
                .ok_or_else(|| "failed to downcast UInt16Builder".to_string())?;
            builder.append_value(value);
            Ok(())
        }),
        DataType::UInt32 => append_unsigned_json(builder, value, "UInt32", |builder, value| {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<UInt32Builder>()
                .ok_or_else(|| "failed to downcast UInt32Builder".to_string())?;
            builder.append_value(value);
            Ok(())
        }),
        DataType::UInt64 => append_unsigned_json(builder, value, "UInt64", |builder, value| {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<UInt64Builder>()
                .ok_or_else(|| "failed to downcast UInt64Builder".to_string())?;
            builder.append_value(value);
            Ok(())
        }),
        DataType::Float16 => {
            Err("NDJSON compaction does not yet support Float16 output".to_string())
        }
        DataType::Float32 => {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<Float32Builder>()
                .ok_or_else(|| "failed to downcast Float32Builder".to_string())?;
            match value {
                Some(Value::Null) | None => builder.append_null(),
                Some(Value::Number(number)) => {
                    let number = number.as_f64().ok_or_else(|| {
                        format!("expected numeric value for Float32, found `{number}`")
                    })?;
                    builder.append_value(number as f32);
                }
                Some(other) => {
                    return Err(format!(
                        "expected numeric value for Float32, found {}",
                        json_kind(other)
                    ));
                }
            }
            Ok(())
        }
        DataType::Float64 => {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<Float64Builder>()
                .ok_or_else(|| "failed to downcast Float64Builder".to_string())?;
            match value {
                Some(Value::Null) | None => builder.append_null(),
                Some(Value::Number(number)) => {
                    let number = number.as_f64().ok_or_else(|| {
                        format!("expected numeric value for Float64, found `{number}`")
                    })?;
                    builder.append_value(number);
                }
                Some(other) => {
                    return Err(format!(
                        "expected numeric value for Float64, found {}",
                        json_kind(other)
                    ));
                }
            }
            Ok(())
        }
        DataType::Utf8 => append_string_json(builder, value, false),
        DataType::LargeUtf8 => append_string_json(builder, value, true),
        DataType::Struct(fields) => {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<StructBuilder>()
                .ok_or_else(|| "failed to downcast StructBuilder".to_string())?;
            match value {
                Some(Value::Object(object)) => {
                    append_struct_from_object(builder, fields, Some(object))
                }
                Some(Value::Null) | None => append_struct_from_object(builder, fields, None),
                Some(other) => Err(format!(
                    "expected object for Struct field, found {}",
                    json_kind(other)
                )),
            }
        }
        DataType::List(field) => {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<ListBuilder<Box<dyn ArrayBuilder>>>()
                .ok_or_else(|| "failed to downcast ListBuilder".to_string())?;
            match value {
                Some(Value::Array(values)) => {
                    for value in values {
                        append_json_value(
                            builder.values().as_mut(),
                            field.data_type(),
                            Some(value),
                        )?;
                    }
                    builder.append(true);
                    Ok(())
                }
                Some(Value::Null) | None => {
                    builder.append(false);
                    Ok(())
                }
                Some(other) => Err(format!(
                    "expected array for List field, found {}",
                    json_kind(other)
                )),
            }
        }
        DataType::LargeList(field) => {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<LargeListBuilder<Box<dyn ArrayBuilder>>>()
                .ok_or_else(|| "failed to downcast LargeListBuilder".to_string())?;
            match value {
                Some(Value::Array(values)) => {
                    for value in values {
                        append_json_value(
                            builder.values().as_mut(),
                            field.data_type(),
                            Some(value),
                        )?;
                    }
                    builder.append(true);
                    Ok(())
                }
                Some(Value::Null) | None => {
                    builder.append(false);
                    Ok(())
                }
                Some(other) => Err(format!(
                    "expected array for LargeList field, found {}",
                    json_kind(other)
                )),
            }
        }
        other => Err(format!(
            "NDJSON compaction does not yet support target type {other:?}"
        )),
    }
}

fn append_signed_json<T, F>(
    builder: &mut dyn ArrayBuilder,
    value: Option<&Value>,
    type_name: &str,
    mut append: F,
) -> Result<(), String>
where
    T: TryFrom<i64>,
    <T as TryFrom<i64>>::Error: std::fmt::Debug,
    F: FnMut(&mut dyn ArrayBuilder, T) -> Result<(), String>,
{
    match value {
        Some(Value::Null) | None => append_null_to_signed(builder, type_name),
        Some(Value::Number(number)) => {
            let number = number.as_i64().ok_or_else(|| {
                format!("expected integer value for {type_name}, found `{number}`")
            })?;
            let converted = T::try_from(number)
                .map_err(|_| format!("value `{number}` is out of range for {type_name}"))?;
            append(builder, converted)
        }
        Some(other) => Err(format!(
            "expected integer value for {type_name}, found {}",
            json_kind(other)
        )),
    }
}

fn append_unsigned_json<T, F>(
    builder: &mut dyn ArrayBuilder,
    value: Option<&Value>,
    type_name: &str,
    mut append: F,
) -> Result<(), String>
where
    T: TryFrom<u64>,
    <T as TryFrom<u64>>::Error: std::fmt::Debug,
    F: FnMut(&mut dyn ArrayBuilder, T) -> Result<(), String>,
{
    match value {
        Some(Value::Null) | None => append_null_to_unsigned(builder, type_name),
        Some(Value::Number(number)) => {
            let number = number.as_u64().ok_or_else(|| {
                format!("expected unsigned integer value for {type_name}, found `{number}`")
            })?;
            let converted = T::try_from(number)
                .map_err(|_| format!("value `{number}` is out of range for {type_name}"))?;
            append(builder, converted)
        }
        Some(other) => Err(format!(
            "expected unsigned integer value for {type_name}, found {}",
            json_kind(other)
        )),
    }
}

fn append_null_to_signed(builder: &mut dyn ArrayBuilder, type_name: &str) -> Result<(), String> {
    match type_name {
        "Int8" => builder
            .as_any_mut()
            .downcast_mut::<Int8Builder>()
            .ok_or_else(|| "failed to downcast Int8Builder".to_string())?
            .append_null(),
        "Int16" => builder
            .as_any_mut()
            .downcast_mut::<Int16Builder>()
            .ok_or_else(|| "failed to downcast Int16Builder".to_string())?
            .append_null(),
        "Int32" => builder
            .as_any_mut()
            .downcast_mut::<Int32Builder>()
            .ok_or_else(|| "failed to downcast Int32Builder".to_string())?
            .append_null(),
        "Int64" => builder
            .as_any_mut()
            .downcast_mut::<Int64Builder>()
            .ok_or_else(|| "failed to downcast Int64Builder".to_string())?
            .append_null(),
        _ => return Err(format!("unsupported signed type `{type_name}`")),
    }
    Ok(())
}

fn append_null_to_unsigned(builder: &mut dyn ArrayBuilder, type_name: &str) -> Result<(), String> {
    match type_name {
        "UInt8" => builder
            .as_any_mut()
            .downcast_mut::<UInt8Builder>()
            .ok_or_else(|| "failed to downcast UInt8Builder".to_string())?
            .append_null(),
        "UInt16" => builder
            .as_any_mut()
            .downcast_mut::<UInt16Builder>()
            .ok_or_else(|| "failed to downcast UInt16Builder".to_string())?
            .append_null(),
        "UInt32" => builder
            .as_any_mut()
            .downcast_mut::<UInt32Builder>()
            .ok_or_else(|| "failed to downcast UInt32Builder".to_string())?
            .append_null(),
        "UInt64" => builder
            .as_any_mut()
            .downcast_mut::<UInt64Builder>()
            .ok_or_else(|| "failed to downcast UInt64Builder".to_string())?
            .append_null(),
        _ => return Err(format!("unsupported unsigned type `{type_name}`")),
    }
    Ok(())
}

fn append_string_json(
    builder: &mut dyn ArrayBuilder,
    value: Option<&Value>,
    is_large: bool,
) -> Result<(), String> {
    match value {
        Some(Value::Null) | None => {
            if is_large {
                builder
                    .as_any_mut()
                    .downcast_mut::<LargeStringBuilder>()
                    .ok_or_else(|| "failed to downcast LargeStringBuilder".to_string())?
                    .append_null();
            } else {
                builder
                    .as_any_mut()
                    .downcast_mut::<StringBuilder>()
                    .ok_or_else(|| "failed to downcast StringBuilder".to_string())?
                    .append_null();
            }
            Ok(())
        }
        Some(value @ (Value::String(_) | Value::Number(_) | Value::Bool(_))) => {
            let rendered = match value {
                Value::String(string) => string.to_string(),
                Value::Number(number) => number.to_string(),
                Value::Bool(boolean) => boolean.to_string(),
                _ => unreachable!(),
            };

            if is_large {
                builder
                    .as_any_mut()
                    .downcast_mut::<LargeStringBuilder>()
                    .ok_or_else(|| "failed to downcast LargeStringBuilder".to_string())?
                    .append_value(rendered);
            } else {
                builder
                    .as_any_mut()
                    .downcast_mut::<StringBuilder>()
                    .ok_or_else(|| "failed to downcast StringBuilder".to_string())?
                    .append_value(rendered);
            }
            Ok(())
        }
        Some(other) => Err(format!(
            "expected primitive or string value for string field, found {}",
            json_kind(other)
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

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn io_error(message: impl Into<String>) -> std::io::Error {
    std::io::Error::other(message.into())
}

fn prefix_payload_conflicts(message: &str) -> String {
    message.replace("field `", "field `payload.")
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
    use serde_json::json;

    fn payload_schema_options() -> PayloadMergeOptions {
        PayloadMergeOptions::default()
    }

    fn compaction_options() -> CompactionOptions {
        CompactionOptions {
            envelope_fields: vec!["event_id".to_string(), "org_id".to_string()],
            payload_column: "payload".to_string(),
            widening_options: WideningOptions::default(),
            batch_rows: 2,
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
