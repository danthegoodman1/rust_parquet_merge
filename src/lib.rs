use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use arrow::array::new_null_array;
use arrow::compute::cast;
use arrow_array::{Array, ArrayRef, LargeListArray, ListArray, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericWideningMode {
    Float64Pragmatic,
    ExactSafe,
}

impl Default for NumericWideningMode {
    fn default() -> Self {
        Self::Float64Pragmatic
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringFallbackMode {
    StrictTyped,
    PrimitiveToStringFallback,
}

impl Default for StringFallbackMode {
    fn default() -> Self {
        Self::StrictTyped
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WideningOptions {
    pub numeric_mode: NumericWideningMode,
    pub string_mode: StringFallbackMode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PayloadMergeOptions {
    pub payload_column: String,
    pub widening_options: WideningOptions,
}

impl Default for PayloadMergeOptions {
    fn default() -> Self {
        Self {
            payload_column: "payload".to_string(),
            widening_options: WideningOptions::default(),
        }
    }
}

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

fn signed_int_width(dt: &DataType) -> Option<u8> {
    match dt {
        DataType::Int8 => Some(8),
        DataType::Int16 => Some(16),
        DataType::Int32 => Some(32),
        DataType::Int64 => Some(64),
        _ => None,
    }
}

fn unsigned_int_width(dt: &DataType) -> Option<u8> {
    match dt {
        DataType::UInt8 => Some(8),
        DataType::UInt16 => Some(16),
        DataType::UInt32 => Some(32),
        DataType::UInt64 => Some(64),
        _ => None,
    }
}

fn signed_type_for_width(width: u8) -> Option<DataType> {
    match width {
        0..=8 => Some(DataType::Int8),
        9..=16 => Some(DataType::Int16),
        17..=32 => Some(DataType::Int32),
        33..=64 => Some(DataType::Int64),
        _ => None,
    }
}

fn unsigned_type_for_width(width: u8) -> Option<DataType> {
    match width {
        0..=8 => Some(DataType::UInt8),
        9..=16 => Some(DataType::UInt16),
        17..=32 => Some(DataType::UInt32),
        33..=64 => Some(DataType::UInt64),
        _ => None,
    }
}

fn exact_safe_numeric_widen(left: &DataType, right: &DataType) -> Option<DataType> {
    if let (Some(left_width), Some(right_width)) = (signed_int_width(left), signed_int_width(right))
    {
        return signed_type_for_width(left_width.max(right_width));
    }

    if let (Some(left_width), Some(right_width)) =
        (unsigned_int_width(left), unsigned_int_width(right))
    {
        return unsigned_type_for_width(left_width.max(right_width));
    }

    if let (Some(signed_width), Some(unsigned_width)) =
        (signed_int_width(left), unsigned_int_width(right))
    {
        return signed_type_for_width(signed_width.max(unsigned_width.saturating_add(1)));
    }

    if let (Some(unsigned_width), Some(signed_width)) =
        (unsigned_int_width(left), signed_int_width(right))
    {
        return signed_type_for_width(signed_width.max(unsigned_width.saturating_add(1)));
    }

    None
}

fn widen_numeric(left: &DataType, right: &DataType, options: &WideningOptions) -> Option<DataType> {
    match options.numeric_mode {
        NumericWideningMode::Float64Pragmatic => {
            if is_float(left) || is_float(right) {
                return Some(DataType::Float64);
            }

            if is_signed_int(left) && is_signed_int(right) {
                return Some(exact_safe_numeric_widen(left, right).expect("signed widen exists"));
            }

            if is_unsigned_int(left) && is_unsigned_int(right) {
                return Some(exact_safe_numeric_widen(left, right).expect("unsigned widen exists"));
            }

            if matches!(left, DataType::UInt64) || matches!(right, DataType::UInt64) {
                return Some(DataType::Float64);
            }

            Some(DataType::Int64)
        }
        NumericWideningMode::ExactSafe => {
            if is_float(left) || is_float(right) {
                return None;
            }
            exact_safe_numeric_widen(left, right)
        }
    }
}

fn typed_string_widen(left: &DataType, right: &DataType) -> Option<DataType> {
    match (left, right) {
        (DataType::Utf8, DataType::LargeUtf8) | (DataType::LargeUtf8, DataType::Utf8) => {
            Some(DataType::LargeUtf8)
        }
        (DataType::Utf8, DataType::Utf8) => Some(DataType::Utf8),
        (DataType::LargeUtf8, DataType::LargeUtf8) => Some(DataType::LargeUtf8),
        _ => None,
    }
}

fn primitive_string_fallback_type(left: &DataType, right: &DataType) -> Option<DataType> {
    if !is_primitive_or_string(left) || !is_primitive_or_string(right) {
        return None;
    }

    if matches!(left, DataType::LargeUtf8) || matches!(right, DataType::LargeUtf8) {
        Some(DataType::LargeUtf8)
    } else {
        Some(DataType::Utf8)
    }
}

fn widen_primitive_or_string(
    left: &DataType,
    right: &DataType,
    options: &WideningOptions,
) -> Option<DataType> {
    if left == right {
        return Some(left.clone());
    }

    match (left, right) {
        (DataType::Null, other) | (other, DataType::Null) => Some(other.clone()),
        _ => typed_string_widen(left, right)
            .or_else(|| {
                if is_numeric(left) && is_numeric(right) {
                    widen_numeric(left, right, options)
                } else {
                    None
                }
            })
            .or_else(|| match options.string_mode {
                StringFallbackMode::StrictTyped => None,
                StringFallbackMode::PrimitiveToStringFallback => {
                    primitive_string_fallback_type(left, right)
                }
            }),
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

fn push_contract_conflict(conflicts: &mut Vec<String>, message: impl Into<String>) {
    conflicts.push(message.into());
}

fn push_metadata_conflict(
    conflicts: &mut Vec<String>,
    path: &str,
    left: &HashMap<String, String>,
    right: &HashMap<String, String>,
) {
    let path_display = if path.is_empty() { "<root>" } else { path };
    conflicts.push(format!(
        "field `{path_display}` incompatible metadata for widening: left={left:?}, right={right:?}"
    ));
}

fn merge_metadata(
    left: &Field,
    right: &Field,
    path: &str,
    conflicts: &mut Vec<String>,
) -> HashMap<String, String> {
    match (left.metadata().is_empty(), right.metadata().is_empty()) {
        (true, true) => HashMap::new(),
        (false, true) => left.metadata().clone(),
        (true, false) => right.metadata().clone(),
        (false, false) if left.metadata() == right.metadata() => left.metadata().clone(),
        (false, false) => {
            push_metadata_conflict(conflicts, path, left.metadata(), right.metadata());
            left.metadata().clone()
        }
    }
}

fn make_merged_field(
    left: &Field,
    right: &Field,
    data_type: DataType,
    nullable: bool,
    path: &str,
    conflicts: &mut Vec<String>,
) -> Field {
    let mut field = Field::new(left.name(), data_type, nullable);
    let metadata = merge_metadata(left, right, path, conflicts);
    if !metadata.is_empty() {
        field = field.with_metadata(metadata);
    }
    field
}

fn widen_fields(
    left: &Field,
    right: &Field,
    path: &str,
    options: &WideningOptions,
    conflicts: &mut Vec<String>,
) -> Field {
    let widened = widen_data_type_with_conflicts(
        path,
        left.data_type(),
        right.data_type(),
        options,
        conflicts,
    );
    let nullable = left.is_nullable() || right.is_nullable();
    make_merged_field(left, right, widened, nullable, path, conflicts)
}

fn widen_struct_fields(
    left: &Fields,
    right: &Fields,
    parent_path: &str,
    options: &WideningOptions,
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
                options,
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
    options: &WideningOptions,
    conflicts: &mut Vec<String>,
) -> FieldRef {
    let element_path = if parent_path.is_empty() {
        "[]".to_string()
    } else {
        format!("{parent_path}[]")
    };
    let widened_child_type = widen_data_type_with_conflicts(
        &element_path,
        left.data_type(),
        right.data_type(),
        options,
        conflicts,
    );
    let nullable = left.is_nullable() || right.is_nullable();
    Arc::new(make_merged_field(
        left,
        right,
        widened_child_type,
        nullable,
        &element_path,
        conflicts,
    ))
}

fn widen_data_type_with_conflicts(
    path: &str,
    left: &DataType,
    right: &DataType,
    options: &WideningOptions,
    conflicts: &mut Vec<String>,
) -> DataType {
    if let Some(widened) = widen_primitive_or_string(left, right, options) {
        return widened;
    }

    match (left, right) {
        (DataType::Struct(left_fields), DataType::Struct(right_fields)) => DataType::Struct(
            widen_struct_fields(left_fields, right_fields, path, options, conflicts),
        ),
        (DataType::List(left_field), DataType::List(right_field)) => DataType::List(
            widen_list_field(left_field, right_field, path, options, conflicts),
        ),
        (DataType::LargeList(left_field), DataType::LargeList(right_field)) => DataType::LargeList(
            widen_list_field(left_field, right_field, path, options, conflicts),
        ),
        (DataType::List(left_field), DataType::LargeList(right_field))
        | (DataType::LargeList(right_field), DataType::List(left_field)) => DataType::LargeList(
            widen_list_field(left_field, right_field, path, options, conflicts),
        ),
        _ => {
            push_conflict(conflicts, path, left, right);
            left.clone()
        }
    }
}

pub fn widen_data_type(
    left: &DataType,
    right: &DataType,
    options: &WideningOptions,
) -> Result<DataType, String> {
    let mut conflicts = Vec::new();
    let widened = widen_data_type_with_conflicts("", left, right, options, &mut conflicts);
    if conflicts.is_empty() {
        Ok(widened)
    } else {
        Err(format_conflicts(&conflicts))
    }
}

pub fn merge_schemas_with_widening(
    left: &Schema,
    right: &Schema,
    options: &WideningOptions,
) -> Result<Schema, String> {
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
                options,
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

fn get_required_field<'a>(
    schema: &'a Schema,
    field_name: &str,
    side: &str,
) -> Result<&'a Field, String> {
    schema
        .field_with_name(field_name)
        .map_err(|_| format!("{side} schema is missing required payload column `{field_name}`"))
}

fn ensure_struct_field<'a>(field: &'a Field, side: &str) -> Result<&'a Fields, String> {
    match field.data_type() {
        DataType::Struct(fields) => Ok(fields),
        other => Err(format!(
            "{side} schema payload column `{}` must be Struct, found {other:?}",
            field.name()
        )),
    }
}

pub fn merge_payload_schemas(
    left: &Schema,
    right: &Schema,
    options: &PayloadMergeOptions,
) -> Result<Schema, String> {
    let payload_column = options.payload_column.as_str();
    let left_payload = get_required_field(left, payload_column, "left")?;
    let right_payload = get_required_field(right, payload_column, "right")?;
    ensure_struct_field(left_payload, "left")?;
    ensure_struct_field(right_payload, "right")?;

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
        let field_name = left_field.name();
        let Some(right_field) = right_by_name.get(field_name.as_str()) else {
            push_contract_conflict(
                &mut conflicts,
                format!("stable envelope column `{field_name}` is missing from right schema"),
            );
            continue;
        };

        if field_name == payload_column {
            merged_fields.push(Arc::new(widen_fields(
                left_field,
                right_field,
                field_name,
                &options.widening_options,
                &mut conflicts,
            )));
            continue;
        }

        if left_field.data_type() != right_field.data_type() {
            push_conflict(
                &mut conflicts,
                field_name,
                left_field.data_type(),
                right_field.data_type(),
            );
            continue;
        }

        let nullable = left_field.is_nullable() || right_field.is_nullable();
        merged_fields.push(Arc::new(make_merged_field(
            left_field,
            right_field,
            left_field.data_type().clone(),
            nullable,
            field_name,
            &mut conflicts,
        )));
    }

    for right_field in right.fields() {
        if !left_names.contains(right_field.name().as_str()) {
            push_contract_conflict(
                &mut conflicts,
                format!(
                    "stable envelope column `{}` is missing from left schema",
                    right_field.name()
                ),
            );
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
pub fn build_index_mapping(source_schema: &Schema, target_schema: &Schema) -> Vec<Option<usize>> {
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
pub fn adjust_with_mapping(
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow_array::{Float32Array, Float64Array, Int32Array, StringArray};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionContext;
    use futures_util::StreamExt;
    use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
    use parquet::arrow::async_writer::AsyncArrowWriter;
    use parquet::file::properties::WriterProperties;
    use tokio::fs::File;
    use tokio::io::{AsyncWrite, BufWriter};

    fn default_options() -> WideningOptions {
        WideningOptions::default()
    }

    fn exact_safe_options() -> WideningOptions {
        WideningOptions {
            numeric_mode: NumericWideningMode::ExactSafe,
            ..WideningOptions::default()
        }
    }

    fn fallback_options() -> WideningOptions {
        WideningOptions {
            string_mode: StringFallbackMode::PrimitiveToStringFallback,
            ..WideningOptions::default()
        }
    }

    fn exact_safe_fallback_options() -> WideningOptions {
        WideningOptions {
            numeric_mode: NumericWideningMode::ExactSafe,
            string_mode: StringFallbackMode::PrimitiveToStringFallback,
        }
    }

    fn unique_parquet_path(prefix: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rust_parquet_merge_{prefix}_{}_{}.parquet",
            std::process::id(),
            nonce
        ))
    }

    async fn create_sample_file(
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

    async fn open_parquet_builder(
        path: &Path,
    ) -> Result<ParquetRecordBatchStreamBuilder<File>, Box<dyn Error>> {
        let file = File::open(path).await?;
        let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
        Ok(builder)
    }

    async fn stream_and_write_from_builder<W>(
        builder: ParquetRecordBatchStreamBuilder<File>,
        writer: &mut AsyncArrowWriter<W>,
        target_schema: &SchemaRef,
        mapping: &[Option<usize>],
    ) -> Result<(), Box<dyn Error>>
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

    async fn merge_payload_files(
        left_path: &Path,
        right_path: &Path,
        output_path: &Path,
        options: &PayloadMergeOptions,
    ) -> Result<SchemaRef, Box<dyn Error>> {
        let (builder_left, builder_right) = tokio::try_join!(
            open_parquet_builder(left_path),
            open_parquet_builder(right_path)
        )?;

        let schema_left = builder_left.schema().clone();
        let schema_right = builder_right.schema().clone();
        let merged_schema = Arc::new(merge_payload_schemas(
            schema_left.as_ref(),
            schema_right.as_ref(),
            options,
        )?);

        let mapping_left = build_index_mapping(schema_left.as_ref(), merged_schema.as_ref());
        let mapping_right = build_index_mapping(schema_right.as_ref(), merged_schema.as_ref());

        let output_file = File::create(output_path).await?;
        let output_file = BufWriter::with_capacity(1 << 20, output_file);
        let mut writer = AsyncArrowWriter::try_new(output_file, merged_schema.clone(), None)?;

        stream_and_write_from_builder(builder_left, &mut writer, &merged_schema, &mapping_left)
            .await?;
        stream_and_write_from_builder(builder_right, &mut writer, &merged_schema, &mapping_right)
            .await?;
        writer.close().await?;

        Ok(merged_schema)
    }

    fn payload_sample_inputs() -> (SchemaRef, RecordBatch, SchemaRef, RecordBatch) {
        let left_profile_fields: Fields =
            vec![Arc::new(Field::new("name", DataType::Utf8, true))].into();
        let left_profile_array = Arc::new(StructArray::new(
            left_profile_fields.clone(),
            vec![Arc::new(StringArray::from(vec![
                Some("Alice"),
                Some("Bob"),
                Some("Charlie"),
            ])) as ArrayRef],
            None,
        )) as ArrayRef;
        let left_payload_fields: Fields = vec![
            Arc::new(Field::new("score", DataType::Int32, true)),
            Arc::new(Field::new(
                "profile",
                DataType::Struct(left_profile_fields.clone()),
                true,
            )),
        ]
        .into();
        let left_payload_array = Arc::new(StructArray::new(
            left_payload_fields.clone(),
            vec![
                Arc::new(Int32Array::from(vec![Some(10), None, Some(30)])) as ArrayRef,
                left_profile_array,
            ],
            None,
        )) as ArrayRef;
        let left_schema = Arc::new(Schema::new(vec![
            Field::new("event_id", DataType::Int32, false),
            Field::new("org_id", DataType::Int32, false),
            Field::new(
                "payload",
                DataType::Struct(left_payload_fields.clone()),
                true,
            ),
        ]));
        let left_batch = RecordBatch::try_new(
            left_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(Int32Array::from(vec![10, 10, 20])) as ArrayRef,
                left_payload_array,
            ],
        )
        .unwrap();

        let right_profile_fields: Fields =
            vec![Arc::new(Field::new("tier", DataType::Utf8, true))].into();
        let right_profile_array = Arc::new(StructArray::new(
            right_profile_fields.clone(),
            vec![Arc::new(StringArray::from(vec![Some("gold"), None])) as ArrayRef],
            None,
        )) as ArrayRef;
        let right_payload_fields: Fields = vec![
            Arc::new(Field::new("score", DataType::Float64, true)),
            Arc::new(Field::new(
                "profile",
                DataType::Struct(right_profile_fields.clone()),
                true,
            )),
        ]
        .into();
        let right_payload_array = Arc::new(StructArray::new(
            right_payload_fields.clone(),
            vec![
                Arc::new(Float64Array::from(vec![Some(44.5), Some(8.25)])) as ArrayRef,
                right_profile_array,
            ],
            None,
        )) as ArrayRef;
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("org_id", DataType::Int32, false),
            Field::new("event_id", DataType::Int32, false),
            Field::new(
                "payload",
                DataType::Struct(right_payload_fields.clone()),
                true,
            ),
        ]));
        let right_batch = RecordBatch::try_new(
            right_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![30, 40])) as ArrayRef,
                Arc::new(Int32Array::from(vec![4, 5])) as ArrayRef,
                right_payload_array,
            ],
        )
        .unwrap();

        (left_schema, left_batch, right_schema, right_batch)
    }

    fn payload_schema_options() -> PayloadMergeOptions {
        PayloadMergeOptions::default()
    }

    fn assert_close(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta < 1e-9, "{left} != {right} (delta={delta})");
    }

    #[test]
    fn widens_numeric_primitive_types() {
        let widened =
            widen_data_type(&DataType::Int32, &DataType::Float32, &default_options()).unwrap();
        assert_eq!(widened, DataType::Float64);
    }

    #[test]
    fn fallback_mode_widens_primitive_to_string_when_one_side_is_utf8() {
        let widened =
            widen_data_type(&DataType::Int32, &DataType::Utf8, &fallback_options()).unwrap();
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

        let widened = widen_data_type(&left, &right, &default_options()).unwrap();
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

        let widened = widen_data_type(&left, &right, &default_options()).unwrap();
        match widened {
            DataType::List(field) => assert_eq!(field.data_type(), &DataType::Float64),
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn widens_mixed_list_and_large_list_to_large_list() {
        let left = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        let right = DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true)));

        let widened = widen_data_type(&left, &right, &default_options()).unwrap();
        match widened {
            DataType::LargeList(field) => assert_eq!(field.data_type(), &DataType::Float64),
            other => panic!("expected LargeList, got {other:?}"),
        }
    }

    #[test]
    fn rejects_primitive_and_struct_conflicts() {
        let struct_type =
            DataType::Struct(vec![Arc::new(Field::new("x", DataType::Int32, true))].into());
        let error = widen_data_type(&DataType::Utf8, &struct_type, &default_options()).unwrap_err();
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

        let error = merge_schemas_with_widening(&left, &right, &default_options()).unwrap_err();
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

        let error = merge_schemas_with_widening(&left, &right, &default_options()).unwrap_err();
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

        let error = merge_schemas_with_widening(&left, &right, &default_options()).unwrap_err();
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
        let list =
            LargeListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(vec![
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

        let merged =
            merge_schemas_with_widening(&schema_left, &schema_right, &default_options()).unwrap();
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

    #[test]
    fn exact_safe_mode_rejects_int_and_float_widening() {
        let error = widen_data_type(&DataType::Int32, &DataType::Float32, &exact_safe_options())
            .unwrap_err();
        assert!(error.contains("Int32"));
        assert!(error.contains("Float32"));
    }

    #[test]
    fn exact_safe_mode_rejects_non_exact_signed_unsigned_mix() {
        let error = widen_data_type(&DataType::Int64, &DataType::UInt64, &exact_safe_options())
            .unwrap_err();
        assert!(error.contains("Int64"));
        assert!(error.contains("UInt64"));
    }

    #[test]
    fn exact_safe_mode_still_allows_exact_integer_widening() {
        let widened =
            widen_data_type(&DataType::Int32, &DataType::UInt8, &exact_safe_options()).unwrap();
        assert_eq!(widened, DataType::Int32);
    }

    #[test]
    fn strict_mode_rejects_numeric_to_string_widening() {
        let error =
            widen_data_type(&DataType::Int32, &DataType::Utf8, &default_options()).unwrap_err();
        assert!(error.contains("Int32"));
        assert!(error.contains("Utf8"));
    }

    #[test]
    fn fallback_mode_widens_boolean_to_string() {
        let widened =
            widen_data_type(&DataType::Boolean, &DataType::Utf8, &fallback_options()).unwrap();
        assert_eq!(widened, DataType::Utf8);
    }

    #[test]
    fn exact_safe_numeric_conflict_can_fall_back_to_string() {
        let widened = widen_data_type(
            &DataType::Int32,
            &DataType::Float32,
            &exact_safe_fallback_options(),
        )
        .unwrap();
        assert_eq!(widened, DataType::Utf8);
    }

    #[test]
    fn typed_string_widening_remains_large_utf8_in_both_modes() {
        let strict =
            widen_data_type(&DataType::Utf8, &DataType::LargeUtf8, &default_options()).unwrap();
        let fallback =
            widen_data_type(&DataType::Utf8, &DataType::LargeUtf8, &fallback_options()).unwrap();
        assert_eq!(strict, DataType::LargeUtf8);
        assert_eq!(fallback, DataType::LargeUtf8);
    }

    #[test]
    fn structural_mismatch_still_conflicts_in_fallback_mode() {
        let struct_type =
            DataType::Struct(vec![Arc::new(Field::new("x", DataType::Int32, true))].into());
        let error =
            widen_data_type(&DataType::Utf8, &struct_type, &fallback_options()).unwrap_err();
        assert!(error.contains("Utf8"));
        assert!(error.contains("Struct"));
    }

    #[test]
    fn merge_schema_preserves_one_sided_metadata() {
        let left = Schema::new(vec![
            Field::new("value", DataType::Int32, true)
                .with_metadata(HashMap::from([("unit".to_string(), "ms".to_string())])),
        ]);
        let right = Schema::new(vec![Field::new("value", DataType::Int32, true)]);

        let merged = merge_schemas_with_widening(&left, &right, &default_options()).unwrap();
        assert_eq!(
            merged
                .field_with_name("value")
                .unwrap()
                .metadata()
                .get("unit"),
            Some(&"ms".to_string())
        );
    }

    #[test]
    fn merge_schema_preserves_identical_metadata() {
        let metadata = HashMap::from([("source".to_string(), "sensor".to_string())]);
        let left = Schema::new(vec![
            Field::new("value", DataType::Int32, true).with_metadata(metadata.clone()),
        ]);
        let right = Schema::new(vec![
            Field::new("value", DataType::Int32, true).with_metadata(metadata),
        ]);

        let merged = merge_schemas_with_widening(&left, &right, &default_options()).unwrap();
        assert_eq!(
            merged
                .field_with_name("value")
                .unwrap()
                .metadata()
                .get("source"),
            Some(&"sensor".to_string())
        );
    }

    #[test]
    fn merge_schema_reports_conflicting_metadata() {
        let left = Schema::new(vec![
            Field::new("value", DataType::Int32, true)
                .with_metadata(HashMap::from([("unit".to_string(), "ms".to_string())])),
        ]);
        let right = Schema::new(vec![
            Field::new("value", DataType::Int32, true)
                .with_metadata(HashMap::from([("unit".to_string(), "s".to_string())])),
        ]);

        let error = merge_schemas_with_widening(&left, &right, &default_options()).unwrap_err();
        assert!(error.contains("field `value`"));
        assert!(error.contains("metadata"));
    }

    #[test]
    fn payload_merge_allows_matching_envelope_columns_with_order_differences() {
        let (left_schema, _, right_schema, _) = payload_sample_inputs();
        let merged = merge_payload_schemas(
            left_schema.as_ref(),
            right_schema.as_ref(),
            &payload_schema_options(),
        )
        .unwrap();

        assert_eq!(merged.fields()[0].name(), "event_id");
        assert_eq!(merged.fields()[1].name(), "org_id");
        assert_eq!(merged.fields()[2].name(), "payload");
        assert_eq!(
            merged.field_with_name("org_id").unwrap().data_type(),
            &DataType::Int32
        );
    }

    #[test]
    fn payload_merge_rejects_envelope_datatype_drift() {
        let left = Schema::new(vec![
            Field::new("event_id", DataType::Int32, false),
            Field::new(
                "payload",
                DataType::Struct(vec![Arc::new(Field::new("score", DataType::Int32, true))].into()),
                true,
            ),
        ]);
        let right = Schema::new(vec![
            Field::new("event_id", DataType::Float64, false),
            Field::new(
                "payload",
                DataType::Struct(vec![Arc::new(Field::new("score", DataType::Int32, true))].into()),
                true,
            ),
        ]);

        let error = merge_payload_schemas(&left, &right, &payload_schema_options()).unwrap_err();
        assert!(error.contains("field `event_id`"));
        assert!(error.contains("Int32"));
        assert!(error.contains("Float64"));
    }

    #[test]
    fn payload_merge_requires_payload_column() {
        let left = Schema::new(vec![
            Field::new("event_id", DataType::Int32, false),
            Field::new(
                "payload",
                DataType::Struct(vec![Arc::new(Field::new("score", DataType::Int32, true))].into()),
                true,
            ),
        ]);
        let right = Schema::new(vec![Field::new("event_id", DataType::Int32, false)]);

        let error = merge_payload_schemas(&left, &right, &payload_schema_options()).unwrap_err();
        assert!(error.contains("missing required payload column `payload`"));
    }

    #[test]
    fn payload_merge_requires_struct_payload_column() {
        let left = Schema::new(vec![
            Field::new("event_id", DataType::Int32, false),
            Field::new("payload", DataType::Utf8, true),
        ]);
        let right = Schema::new(vec![
            Field::new("event_id", DataType::Int32, false),
            Field::new(
                "payload",
                DataType::Struct(vec![Arc::new(Field::new("score", DataType::Int32, true))].into()),
                true,
            ),
        ]);

        let error = merge_payload_schemas(&left, &right, &payload_schema_options()).unwrap_err();
        assert!(error.contains("payload column `payload` must be Struct"));
    }

    #[test]
    fn payload_merge_widens_nested_payload_fields() {
        let (left_schema, _, right_schema, _) = payload_sample_inputs();
        let merged = merge_payload_schemas(
            left_schema.as_ref(),
            right_schema.as_ref(),
            &payload_schema_options(),
        )
        .unwrap();

        let payload_type = merged.field_with_name("payload").unwrap().data_type();
        match payload_type {
            DataType::Struct(fields) => {
                assert_eq!(fields[0].name(), "score");
                assert_eq!(fields[0].data_type(), &DataType::Float64);
                assert_eq!(fields[1].name(), "profile");
                match fields[1].data_type() {
                    DataType::Struct(profile_fields) => {
                        assert_eq!(profile_fields[0].name(), "name");
                        assert_eq!(profile_fields[1].name(), "tier");
                    }
                    other => panic!("expected Struct payload.profile, got {other:?}"),
                }
            }
            other => panic!("expected Struct payload, got {other:?}"),
        }
    }

    #[test]
    fn payload_merge_rejects_incompatible_nested_payload_fields() {
        let left = Schema::new(vec![
            Field::new("event_id", DataType::Int32, false),
            Field::new(
                "payload",
                DataType::Struct(
                    vec![Arc::new(Field::new("profile", DataType::Int32, true))].into(),
                ),
                true,
            ),
        ]);
        let right = Schema::new(vec![
            Field::new("event_id", DataType::Int32, false),
            Field::new(
                "payload",
                DataType::Struct(
                    vec![Arc::new(Field::new(
                        "profile",
                        DataType::Struct(
                            vec![Arc::new(Field::new("name", DataType::Utf8, true))].into(),
                        ),
                        true,
                    ))]
                    .into(),
                ),
                true,
            ),
        ]);

        let error = merge_payload_schemas(&left, &right, &payload_schema_options()).unwrap_err();
        assert!(error.contains("field `payload.profile`"));
        assert!(error.contains("Int32"));
        assert!(error.contains("Struct"));
    }

    #[tokio::test]
    async fn payload_merge_end_to_end_writes_typed_merged_output() -> Result<(), Box<dyn Error>> {
        let (left_schema, left_batch, right_schema, right_batch) = payload_sample_inputs();
        let left_path = unique_parquet_path("payload_left");
        let right_path = unique_parquet_path("payload_right");
        let output_path = unique_parquet_path("payload_merged");

        create_sample_file(&left_path, left_schema, left_batch).await?;
        create_sample_file(&right_path, right_schema, right_batch).await?;

        let merged_schema = merge_payload_files(
            &left_path,
            &right_path,
            &output_path,
            &payload_schema_options(),
        )
        .await?;
        let merged_batches = read_parquet_batches(&output_path).await?;
        let merged_batch = merged_batches
            .first()
            .expect("merged parquet contains one output batch");

        assert_eq!(merged_schema.field(0).name(), "event_id");
        assert_eq!(merged_schema.field(1).name(), "org_id");
        assert_eq!(merged_schema.field(2).name(), "payload");

        let payload_array = merged_batch
            .column(merged_batch.schema().index_of("payload")?)
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("payload column is a StructArray");
        let payload_fields = match payload_array.data_type() {
            DataType::Struct(fields) => fields,
            other => panic!("expected payload Struct type, got {other:?}"),
        };
        let score_index = payload_fields
            .iter()
            .position(|field| field.name() == "score")
            .expect("payload.score exists");
        let profile_index = payload_fields
            .iter()
            .position(|field| field.name() == "profile")
            .expect("payload.profile exists");

        let score_array = payload_array
            .column(score_index)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("payload.score is Float64");
        assert_close(score_array.value(0), 10.0);
        assert!(score_array.is_null(1));
        assert_close(score_array.value(2), 30.0);
        assert_close(score_array.value(3), 44.5);
        assert_close(score_array.value(4), 8.25);

        let profile_array = payload_array
            .column(profile_index)
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("payload.profile is a StructArray");
        let profile_fields = match profile_array.data_type() {
            DataType::Struct(fields) => fields,
            other => panic!("expected payload.profile Struct type, got {other:?}"),
        };
        let name_index = profile_fields
            .iter()
            .position(|field| field.name() == "name")
            .expect("payload.profile.name exists");
        let tier_index = profile_fields
            .iter()
            .position(|field| field.name() == "tier")
            .expect("payload.profile.tier exists");

        let name_array = profile_array
            .column(name_index)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("payload.profile.name is Utf8");
        assert_eq!(name_array.value(0), "Alice");
        assert_eq!(name_array.value(1), "Bob");
        assert_eq!(name_array.value(2), "Charlie");
        assert!(name_array.is_null(3));
        assert!(name_array.is_null(4));

        let tier_array = profile_array
            .column(tier_index)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("payload.profile.tier is Utf8");
        assert!(tier_array.is_null(0));
        assert!(tier_array.is_null(1));
        assert!(tier_array.is_null(2));
        assert_eq!(tier_array.value(3), "gold");
        assert!(tier_array.is_null(4));

        let _ = tokio::fs::remove_file(&left_path).await;
        let _ = tokio::fs::remove_file(&right_path).await;
        let _ = tokio::fs::remove_file(&output_path).await;

        Ok(())
    }

    #[tokio::test]
    async fn datafusion_can_query_typed_nested_payload_fields() -> Result<(), Box<dyn Error>> {
        let (left_schema, left_batch, right_schema, right_batch) = payload_sample_inputs();
        let left_path = unique_parquet_path("df_payload_left");
        let right_path = unique_parquet_path("df_payload_right");
        let output_path = unique_parquet_path("df_payload_merged");

        create_sample_file(&left_path, left_schema, left_batch).await?;
        create_sample_file(&right_path, right_schema, right_batch).await?;
        merge_payload_files(
            &left_path,
            &right_path,
            &output_path,
            &payload_schema_options(),
        )
        .await?;

        let merged_batches = read_parquet_batches(&output_path).await?;
        let merged_schema = merged_batches[0].schema();
        let ctx = SessionContext::new();
        let mem_table = MemTable::try_new(merged_schema, vec![merged_batches.clone()])?;
        ctx.register_table("merged_payload", Arc::new(mem_table))?;

        let aggregate_batches = ctx
            .sql("SELECT sum(payload['score']) AS total_score FROM merged_payload")
            .await?
            .collect()
            .await?;
        let aggregate_batch = aggregate_batches
            .first()
            .expect("aggregate query returns one batch");
        let total_score = aggregate_batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("aggregate result is Float64")
            .value(0);
        assert_close(total_score, 92.75);

        let projection_batches = ctx
            .sql(
                "SELECT event_id, payload['profile']['name'] AS profile_name \
                 FROM merged_payload ORDER BY event_id",
            )
            .await?
            .collect()
            .await?;
        let projection_batch = projection_batches
            .first()
            .expect("projection query returns one batch");
        let event_ids = projection_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("event_id projection is Int32");
        let profile_names = projection_batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("profile_name projection is Utf8");

        assert_eq!(event_ids.values(), &[1, 2, 3, 4, 5]);
        assert_eq!(profile_names.value(0), "Alice");
        assert_eq!(profile_names.value(1), "Bob");
        assert_eq!(profile_names.value(2), "Charlie");
        assert!(profile_names.is_null(3));
        assert!(profile_names.is_null(4));

        let _ = tokio::fs::remove_file(&left_path).await;
        let _ = tokio::fs::remove_file(&right_path).await;
        let _ = tokio::fs::remove_file(&output_path).await;

        Ok(())
    }
}
