use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::new_null_array;
use arrow::compute::cast;
use arrow_array::{
    Array, ArrayRef, Float64Array, Int32Array, LargeListArray, ListArray, RecordBatch,
    StringArray, StructArray,
};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use futures_util::StreamExt;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_writer::AsyncArrowWriter;
use parquet::file::properties::WriterProperties;
use tokio::fs::File;
use tokio::io::{AsyncWrite, BufWriter};

fn to_parquet_error(message: impl Into<String>) -> parquet::errors::ParquetError {
    parquet::errors::ParquetError::General(message.into())
}

fn is_signed_int(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
    )
}

fn is_unsigned_int(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
    )
}

fn is_float(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Float16 | DataType::Float32 | DataType::Float64
    )
}

fn is_numeric(dt: &DataType) -> bool {
    is_signed_int(dt) || is_unsigned_int(dt) || is_float(dt)
}

fn is_primitive_or_string(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Null
            | DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
    )
}

fn widen_numeric(left: &DataType, right: &DataType) -> DataType {
    if is_float(left) || is_float(right) {
        return DataType::Float64;
    }

    if is_signed_int(left) && is_signed_int(right) {
        if matches!(left, DataType::Int64) || matches!(right, DataType::Int64) {
            return DataType::Int64;
        }
        if matches!(left, DataType::Int32) || matches!(right, DataType::Int32) {
            return DataType::Int32;
        }
        if matches!(left, DataType::Int16) || matches!(right, DataType::Int16) {
            return DataType::Int16;
        }
        return DataType::Int8;
    }

    if is_unsigned_int(left) && is_unsigned_int(right) {
        if matches!(left, DataType::UInt64) || matches!(right, DataType::UInt64) {
            return DataType::UInt64;
        }
        if matches!(left, DataType::UInt32) || matches!(right, DataType::UInt32) {
            return DataType::UInt32;
        }
        if matches!(left, DataType::UInt16) || matches!(right, DataType::UInt16) {
            return DataType::UInt16;
        }
        return DataType::UInt8;
    }

    if matches!(left, DataType::UInt64) || matches!(right, DataType::UInt64) {
        return DataType::Float64;
    }

    DataType::Int64
}

fn widen_primitive_or_string(left: &DataType, right: &DataType) -> Option<DataType> {
    if left == right {
        return Some(left.clone());
    }

    match (left, right) {
        (DataType::Null, other) | (other, DataType::Null) => Some(other.clone()),
        (DataType::Utf8, DataType::LargeUtf8) | (DataType::LargeUtf8, DataType::Utf8) => {
            Some(DataType::LargeUtf8)
        }
        (DataType::Utf8, other) | (other, DataType::Utf8) if is_primitive_or_string(other) => {
            Some(DataType::Utf8)
        }
        (DataType::LargeUtf8, other) | (other, DataType::LargeUtf8)
            if is_primitive_or_string(other) =>
        {
            Some(DataType::LargeUtf8)
        }
        (left_num, right_num) if is_numeric(left_num) && is_numeric(right_num) => {
            Some(widen_numeric(left_num, right_num))
        }
        (DataType::Boolean, other) | (other, DataType::Boolean)
            if is_primitive_or_string(other) =>
        {
            Some(DataType::Utf8)
        }
        _ => None,
    }
}

fn make_nullable(field: &Field, nullable: bool) -> Field {
    if field.is_nullable() == nullable {
        return field.clone();
    }
    let mut new_field = Field::new(field.name(), field.data_type().clone(), nullable);
    if !field.metadata().is_empty() {
        new_field = new_field.with_metadata(field.metadata().clone());
    }
    new_field
}

fn format_conflicts(conflicts: &[String]) -> String {
    format!(
        "schema widening found {} conflict(s):\n- {}",
        conflicts.len(),
        conflicts.join("\n- ")
    )
}

fn push_conflict(conflicts: &mut Vec<String>, path: &str, left: &DataType, right: &DataType) {
    let path_display = if path.is_empty() { "<root>" } else { path };
    conflicts.push(format!(
        "field `{path_display}` incompatible types for widening: left={left:?}, right={right:?}"
    ));
}

fn widen_fields(left: &Field, right: &Field, path: &str, conflicts: &mut Vec<String>) -> Field {
    let widened = widen_data_type_with_conflicts(path, left.data_type(), right.data_type(), conflicts);
    let nullable = left.is_nullable() || right.is_nullable();
    Field::new(left.name(), widened, nullable)
}

fn widen_struct_fields(
    left: &Fields,
    right: &Fields,
    parent_path: &str,
    conflicts: &mut Vec<String>,
) -> Fields {
    let right_by_name: HashMap<&str, &FieldRef> = right
        .iter()
        .map(|field| (field.name().as_str(), field))
        .collect();
    let left_names: BTreeSet<&str> = left.iter().map(|field| field.name().as_str()).collect();

    let mut widened: Vec<FieldRef> = Vec::new();

    for left_field in left {
        let child_path = if parent_path.is_empty() {
            left_field.name().to_string()
        } else {
            format!("{parent_path}.{}", left_field.name())
        };

        if let Some(right_field) = right_by_name.get(left_field.name().as_str()) {
            widened.push(Arc::new(widen_fields(
                left_field,
                right_field,
                &child_path,
                conflicts,
            )));
        } else {
            widened.push(Arc::new(make_nullable(left_field, true)));
        }
    }

    for right_field in right {
        if !left_names.contains(right_field.name().as_str()) {
            widened.push(Arc::new(make_nullable(right_field, true)));
        }
    }

    widened.into()
}

fn widen_list_field(
    left: &FieldRef,
    right: &FieldRef,
    parent_path: &str,
    conflicts: &mut Vec<String>,
) -> FieldRef {
    let element_path = if parent_path.is_empty() {
        "[]".to_string()
    } else {
        format!("{parent_path}[]")
    };
    let widened_child_type =
        widen_data_type_with_conflicts(&element_path, left.data_type(), right.data_type(), conflicts);
    let nullable = left.is_nullable() || right.is_nullable();
    Arc::new(Field::new(left.name(), widened_child_type, nullable))
}

fn widen_data_type_with_conflicts(
    path: &str,
    left: &DataType,
    right: &DataType,
    conflicts: &mut Vec<String>,
) -> DataType {
    if let Some(widened) = widen_primitive_or_string(left, right) {
        return widened;
    }

    match (left, right) {
        (DataType::Struct(left_fields), DataType::Struct(right_fields)) => {
            DataType::Struct(widen_struct_fields(left_fields, right_fields, path, conflicts))
        }
        (DataType::List(left_field), DataType::List(right_field)) => {
            DataType::List(widen_list_field(left_field, right_field, path, conflicts))
        }
        (DataType::LargeList(left_field), DataType::LargeList(right_field)) => {
            DataType::LargeList(widen_list_field(left_field, right_field, path, conflicts))
        }
        (DataType::List(left_field), DataType::LargeList(right_field))
        | (DataType::LargeList(right_field), DataType::List(left_field)) => {
            DataType::LargeList(widen_list_field(left_field, right_field, path, conflicts))
        }
        _ => {
            push_conflict(conflicts, path, left, right);
            left.clone()
        }
    }
}

fn widen_data_type(left: &DataType, right: &DataType) -> Result<DataType, String> {
    let mut conflicts = Vec::new();
    let widened = widen_data_type_with_conflicts("", left, right, &mut conflicts);
    if conflicts.is_empty() {
        Ok(widened)
    } else {
        Err(format_conflicts(&conflicts))
    }
}

fn merge_schemas_with_widening(left: &Schema, right: &Schema) -> Result<Schema, String> {
    let right_by_name: HashMap<&str, &FieldRef> = right
        .fields()
        .iter()
        .map(|field| (field.name().as_str(), field))
        .collect();
    let left_names: BTreeSet<&str> = left
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect();

    let mut merged_fields: Vec<FieldRef> = Vec::new();
    let mut conflicts = Vec::new();

    for left_field in left.fields() {
        if let Some(right_field) = right_by_name.get(left_field.name().as_str()) {
            merged_fields.push(Arc::new(widen_fields(
                left_field,
                right_field,
                left_field.name(),
                &mut conflicts,
            )));
        } else {
            merged_fields.push(Arc::new(make_nullable(left_field, true)));
        }
    }

    for right_field in right.fields() {
        if !left_names.contains(right_field.name().as_str()) {
            merged_fields.push(Arc::new(make_nullable(right_field, true)));
        }
    }

    if conflicts.is_empty() {
        Ok(Schema::new(merged_fields))
    } else {
        Err(format_conflicts(&conflicts))
    }
}

/// Build a one-time column index mapping from a source schema to a target schema.
/// Each entry is `Some(source_index)` if the target field exists in the source, otherwise `None`.
fn build_index_mapping(source_schema: &Schema, target_schema: &Schema) -> Vec<Option<usize>> {
    target_schema
        .fields()
        .iter()
        .map(|target_field| source_schema.index_of(target_field.name()).ok())
        .collect()
}

fn coerce_struct_array(array: &ArrayRef, target_fields: &Fields) -> Result<ArrayRef, String> {
    let source_struct = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| format!("expected StructArray, got {:?}", array.data_type()))?;

    let source_field_lookup = match array.data_type() {
        DataType::Struct(fields) => fields
            .iter()
            .enumerate()
            .map(|(index, field)| (field.name().clone(), index))
            .collect::<HashMap<String, usize>>(),
        other => {
            return Err(format!(
                "coerce_struct_array called with non-struct type: {other:?}"
            ));
        }
    };

    let mut coerced_children = Vec::with_capacity(target_fields.len());
    for target_child in target_fields {
        if let Some(source_index) = source_field_lookup.get(target_child.name().as_str()) {
            coerced_children.push(coerce_array_to_type(
                source_struct.column(*source_index),
                target_child.data_type(),
            )?);
        } else {
            coerced_children.push(new_null_array(
                target_child.data_type(),
                source_struct.len(),
            ));
        }
    }

    Ok(Arc::new(StructArray::new(
        target_fields.clone(),
        coerced_children,
        source_struct.nulls().cloned(),
    )))
}

fn coerce_list_array(array: &ArrayRef, target_field: &FieldRef) -> Result<ArrayRef, String> {
    let source_list = array
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| format!("expected ListArray, got {:?}", array.data_type()))?;

    let coerced_values = coerce_array_to_type(&source_list.values(), target_field.data_type())?;
    let target_list_field = Arc::new(Field::new(
        target_field.name(),
        target_field.data_type().clone(),
        target_field.is_nullable(),
    ));

    Ok(Arc::new(ListArray::new(
        target_list_field,
        source_list.offsets().clone(),
        coerced_values,
        source_list.nulls().cloned(),
    )))
}

fn coerce_large_list_array(array: &ArrayRef, target_field: &FieldRef) -> Result<ArrayRef, String> {
    let source_list = array
        .as_any()
        .downcast_ref::<LargeListArray>()
        .ok_or_else(|| format!("expected LargeListArray, got {:?}", array.data_type()))?;

    let coerced_values = coerce_array_to_type(&source_list.values(), target_field.data_type())?;
    let target_list_field = Arc::new(Field::new(
        target_field.name(),
        target_field.data_type().clone(),
        target_field.is_nullable(),
    ));

    Ok(Arc::new(LargeListArray::new(
        target_list_field,
        source_list.offsets().clone(),
        coerced_values,
        source_list.nulls().cloned(),
    )))
}

fn coerce_list_like_array(array: &ArrayRef, target_type: &DataType) -> Result<ArrayRef, String> {
    match (array.data_type(), target_type) {
        (DataType::List(_), DataType::List(target_field)) => coerce_list_array(array, target_field),
        (DataType::LargeList(_), DataType::LargeList(target_field)) => {
            coerce_large_list_array(array, target_field)
        }
        (DataType::List(_), DataType::LargeList(target_field)) => {
            let source_list = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| format!("expected ListArray, got {:?}", array.data_type()))?;

            let coerced_values =
                coerce_array_to_type(&source_list.values(), target_field.data_type())?;
            let intermediate_field = Arc::new(Field::new(
                target_field.name(),
                target_field.data_type().clone(),
                target_field.is_nullable(),
            ));
            let intermediate = Arc::new(ListArray::new(
                intermediate_field,
                source_list.offsets().clone(),
                coerced_values,
                source_list.nulls().cloned(),
            )) as ArrayRef;

            cast(intermediate.as_ref(), target_type).map_err(|error| {
                format!(
                    "failed list-to-large-list cast from {:?} to {:?}: {error}",
                    array.data_type(),
                    target_type
                )
            })
        }
        (DataType::LargeList(_), DataType::List(target_field)) => {
            let source_list = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| format!("expected LargeListArray, got {:?}", array.data_type()))?;

            let coerced_values =
                coerce_array_to_type(&source_list.values(), target_field.data_type())?;
            let intermediate_field = Arc::new(Field::new(
                target_field.name(),
                target_field.data_type().clone(),
                target_field.is_nullable(),
            ));
            let intermediate = Arc::new(LargeListArray::new(
                intermediate_field,
                source_list.offsets().clone(),
                coerced_values,
                source_list.nulls().cloned(),
            )) as ArrayRef;

            cast(intermediate.as_ref(), target_type).map_err(|error| {
                format!(
                    "failed large-list-to-list cast from {:?} to {:?}: {error}",
                    array.data_type(),
                    target_type
                )
            })
        }
        _ => Err(format!(
            "unsupported list coercion from {:?} to {:?}",
            array.data_type(),
            target_type
        )),
    }
}

fn coerce_array_to_type(array: &ArrayRef, target_type: &DataType) -> Result<ArrayRef, String> {
    if array.data_type() == target_type {
        return Ok(array.clone());
    }

    match (array.data_type(), target_type) {
        (DataType::Struct(_), DataType::Struct(target_fields)) => {
            coerce_struct_array(array, target_fields)
        }
        (DataType::List(_), DataType::List(_))
        | (DataType::List(_), DataType::LargeList(_))
        | (DataType::LargeList(_), DataType::List(_))
        | (DataType::LargeList(_), DataType::LargeList(_)) => {
            coerce_list_like_array(array, target_type)
        }
        (source, target) if is_primitive_or_string(source) && is_primitive_or_string(target) => {
            cast(array, target_type).map_err(|error| {
                format!("failed primitive cast from {source:?} to {target:?}: {error}")
            })
        }
        (source, target) => Err(format!(
            "unsupported coercion from {source:?} to {target:?}"
        )),
    }
}

/// Adjust a `RecordBatch` to match `target_schema` using a precomputed index mapping.
fn adjust_with_mapping(
    batch: &RecordBatch,
    target_schema: &SchemaRef,
    mapping: &[Option<usize>],
) -> Result<RecordBatch, parquet::errors::ParquetError> {
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(mapping.len());

    for (i, maybe_src_idx) in mapping.iter().enumerate() {
        match maybe_src_idx {
            Some(src_idx) => {
                let target_type = target_schema.field(i).data_type();
                let coerced = coerce_array_to_type(batch.column(*src_idx), target_type)
                    .map_err(to_parquet_error)?;
                new_columns.push(coerced);
            }
            None => new_columns.push(new_null_array(
                target_schema.field(i).data_type(),
                batch.num_rows(),
            )),
        }
    }

    RecordBatch::try_new(target_schema.clone(), new_columns)
        .map_err(|error| to_parquet_error(format!("Arrow error: {error}")))
}

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
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30)])),
            Arc::new(StringArray::from(vec![
                Some("Alice"),
                Some("Bob"),
                Some("Charlie"),
            ])),
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
            Arc::new(Int32Array::from(vec![4, 5])),
            Arc::new(Float64Array::from(vec![Some(44.5), Some(8.25)])),
            Arc::new(Int32Array::from(vec![Some(30), None])),
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
    let merged_schema = Arc::new(merge_schemas_with_widening(
        schema1.as_ref(),
        schema2.as_ref(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float32Array;

    #[test]
    fn widens_numeric_primitive_types() {
        let widened = widen_data_type(&DataType::Int32, &DataType::Float32).unwrap();
        assert_eq!(widened, DataType::Float64);
    }

    #[test]
    fn widens_primitive_to_string_when_one_side_is_utf8() {
        let widened = widen_data_type(&DataType::Int32, &DataType::Utf8).unwrap();
        assert_eq!(widened, DataType::Utf8);
    }

    #[test]
    fn widens_struct_fields_recursively() {
        let left = DataType::Struct(
            vec![
                Arc::new(Field::new("a", DataType::Int32, false)),
                Arc::new(Field::new("b", DataType::Utf8, true)),
            ]
            .into(),
        );
        let right = DataType::Struct(
            vec![
                Arc::new(Field::new("a", DataType::Float64, false)),
                Arc::new(Field::new("c", DataType::Int32, false)),
            ]
            .into(),
        );

        let widened = widen_data_type(&left, &right).unwrap();
        match widened {
            DataType::Struct(fields) => {
                assert_eq!(fields[0].name(), "a");
                assert_eq!(fields[0].data_type(), &DataType::Float64);
                assert_eq!(fields[1].name(), "b");
                assert!(fields[1].is_nullable());
                assert_eq!(fields[2].name(), "c");
                assert!(fields[2].is_nullable());
            }
            other => panic!("expected Struct, got {other:?}"),
        }
    }

    #[test]
    fn widens_list_child_type_recursively() {
        let left = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        let right = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));

        let widened = widen_data_type(&left, &right).unwrap();
        match widened {
            DataType::List(field) => assert_eq!(field.data_type(), &DataType::Float64),
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn widens_mixed_list_and_large_list_to_large_list() {
        let left = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        let right = DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true)));

        let widened = widen_data_type(&left, &right).unwrap();
        match widened {
            DataType::LargeList(field) => assert_eq!(field.data_type(), &DataType::Float64),
            other => panic!("expected LargeList, got {other:?}"),
        }
    }

    #[test]
    fn rejects_primitive_and_struct_conflicts() {
        let struct_type =
            DataType::Struct(vec![Arc::new(Field::new("x", DataType::Int32, true))].into());
        let error = widen_data_type(&DataType::Utf8, &struct_type).unwrap_err();
        assert!(error.contains("incompatible types"));
        assert!(error.contains("Utf8"));
        assert!(error.contains("Struct"));
    }

    #[test]
    fn conflict_message_includes_top_level_property_and_types() {
        let left = Schema::new(vec![Field::new("payload", DataType::Utf8, true)]);
        let right = Schema::new(vec![Field::new(
            "payload",
            DataType::Struct(vec![Arc::new(Field::new("x", DataType::Int32, true))].into()),
            true,
        )]);

        let error = merge_schemas_with_widening(&left, &right).unwrap_err();
        assert!(error.contains("field `payload`"));
        assert!(error.contains("Utf8"));
        assert!(error.contains("Struct"));
    }

    #[test]
    fn conflict_message_includes_nested_property_and_types() {
        let left = Schema::new(vec![Field::new(
            "payload",
            DataType::Struct(vec![Arc::new(Field::new("x", DataType::Int32, true))].into()),
            true,
        )]);
        let right = Schema::new(vec![Field::new(
            "payload",
            DataType::Struct(
                vec![Arc::new(Field::new(
                    "x",
                    DataType::Struct(vec![Arc::new(Field::new("y", DataType::Utf8, true))].into()),
                    true,
                ))]
                .into(),
            ),
            true,
        )]);

        let error = merge_schemas_with_widening(&left, &right).unwrap_err();
        assert!(error.contains("field `payload.x`"));
        assert!(error.contains("Int32"));
        assert!(error.contains("Struct"));
    }

    #[test]
    fn conflict_message_collects_multiple_conflicts() {
        let left = Schema::new(vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Int32, true),
        ]);
        let right = Schema::new(vec![
            Field::new(
                "a",
                DataType::Struct(vec![Arc::new(Field::new("x", DataType::Int32, true))].into()),
                true,
            ),
            Field::new(
                "b",
                DataType::Struct(vec![Arc::new(Field::new("y", DataType::Utf8, true))].into()),
                true,
            ),
        ]);

        let error = merge_schemas_with_widening(&left, &right).unwrap_err();
        assert!(error.contains("2 conflict(s)"));
        assert!(error.contains("field `a`"));
        assert!(error.contains("field `b`"));
    }

    #[test]
    fn adjust_with_mapping_casts_primitive_columns() {
        let source_schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]));
        let batch = RecordBatch::try_new(
            source_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![Some(1), Some(2), None]))],
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, true),
            Field::new("missing", DataType::Utf8, true),
        ]));
        let mapping = build_index_mapping(source_schema.as_ref(), target_schema.as_ref());
        let adjusted = adjust_with_mapping(&batch, &target_schema, &mapping).unwrap();

        assert_eq!(adjusted.schema().field(0).data_type(), &DataType::Float64);
        assert_eq!(adjusted.schema().field(1).data_type(), &DataType::Utf8);
        assert_eq!(adjusted.column(1).null_count(), adjusted.num_rows());
    }

    #[test]
    fn coerce_struct_array_adds_missing_fields() {
        let source_struct = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("a", DataType::Int32, true)),
            Arc::new(Int32Array::from(vec![Some(1), None])) as ArrayRef,
        )])) as ArrayRef;

        let target_type = DataType::Struct(
            vec![
                Arc::new(Field::new("a", DataType::Float64, true)),
                Arc::new(Field::new("b", DataType::Utf8, true)),
            ]
            .into(),
        );
        let coerced = coerce_array_to_type(&source_struct, &target_type).unwrap();
        assert_eq!(coerced.data_type(), &target_type);
    }

    #[test]
    fn coerce_list_array_casts_child_values() {
        let list = ListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(3)]),
            None,
        ]);
        let source = Arc::new(list) as ArrayRef;
        let target_type = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
        let coerced = coerce_array_to_type(&source, &target_type).unwrap();
        assert_eq!(coerced.data_type(), &target_type);

        let values = coerced
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .values();
        assert_eq!(values.data_type(), &DataType::Float64);
    }

    #[test]
    fn coerce_list_to_large_list_casts_child_values() {
        let list = ListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(3)]),
            None,
        ]);
        let source = Arc::new(list) as ArrayRef;
        let target_type =
            DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true)));
        let coerced = coerce_array_to_type(&source, &target_type).unwrap();
        assert_eq!(coerced.data_type(), &target_type);

        let values = coerced
            .as_any()
            .downcast_ref::<LargeListArray>()
            .unwrap()
            .values();
        assert_eq!(values.data_type(), &DataType::Float64);
    }

    #[test]
    fn coerce_large_list_to_list_casts_child_values() {
        let list = LargeListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(
            vec![Some(vec![Some(1), Some(2)]), Some(vec![Some(3)]), None],
        );
        let source = Arc::new(list) as ArrayRef;
        let target_type = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
        let coerced = coerce_array_to_type(&source, &target_type).unwrap();
        assert_eq!(coerced.data_type(), &target_type);

        let values = coerced
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .values();
        assert_eq!(values.data_type(), &DataType::Float64);
    }

    #[test]
    fn merge_schema_with_widening_uses_promoted_type_for_shared_columns() {
        let schema_left = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("score", DataType::Int32, true),
        ]);
        let schema_right = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("score", DataType::Float64, true),
            Field::new("name", DataType::Utf8, true),
        ]);

        let merged = merge_schemas_with_widening(&schema_left, &schema_right).unwrap();
        assert_eq!(
            merged.field_with_name("score").unwrap().data_type(),
            &DataType::Float64
        );
        assert_eq!(
            merged.field_with_name("name").unwrap().data_type(),
            &DataType::Utf8
        );
    }

    #[test]
    fn widening_handles_float32_and_int32_in_batch_cast() {
        let source_schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float32,
            true,
        )]));
        let batch = RecordBatch::try_new(
            source_schema.clone(),
            vec![Arc::new(Float32Array::from(vec![
                Some(1.0),
                None,
                Some(2.5),
            ]))],
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let mapping = build_index_mapping(source_schema.as_ref(), target_schema.as_ref());
        let adjusted = adjust_with_mapping(&batch, &target_schema, &mapping).unwrap();
        assert_eq!(adjusted.column(0).data_type(), &DataType::Float64);
    }
}
