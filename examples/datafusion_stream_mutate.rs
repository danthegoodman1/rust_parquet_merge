use arrow::{
    array::{
        Array, ArrayRef, RecordBatch, StringArray, StructArray, TimestampMillisecondArray,
        UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    util::display::array_value_to_string,
};
use chrono::{DateTime, Utc};
use datafusion::{datasource::MemTable, prelude::*};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// Sample JSON structure for our events
#[derive(Debug, Serialize, Deserialize)]
struct Event {
    timestamp: DateTime<Utc>,
    org_id: u64,
    user_id: String,
    event_type: String,
    amount: Option<f64>,
    country: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Create sample JSON data
    let json_data = vec![
        Event {
            timestamp: "2025-01-15T10:30:00Z".parse()?,
            org_id: 100,
            user_id: "user123".to_string(),
            event_type: "purchase".to_string(),
            amount: Some(99.99),
            country: "US".to_string(),
        },
        Event {
            timestamp: "2025-01-15T14:45:00Z".parse()?,
            org_id: 100,
            user_id: "user456".to_string(),
            event_type: "signup".to_string(),
            amount: None,
            country: "US".to_string(),
        },
        Event {
            timestamp: "2025-01-16T08:15:00Z".parse()?,
            org_id: 200,
            user_id: "user789".to_string(),
            event_type: "purchase".to_string(),
            amount: Some(149.99),
            country: "UK".to_string(),
        },
        Event {
            timestamp: "2025-01-16T12:00:00Z".parse()?,
            org_id: 300,
            user_id: "user111".to_string(),
            event_type: "login".to_string(),
            amount: None,
            country: "CA".to_string(),
        },
        Event {
            timestamp: "2025-01-17T09:30:00Z".parse()?,
            org_id: 200,
            user_id: "user222".to_string(),
            event_type: "purchase".to_string(),
            amount: Some(79.99),
            country: "UK".to_string(),
        },
    ];

    println!("=== Original JSON Data ===");
    for event in &json_data {
        println!("{:?}", event);
    }
    println!();

    // Step 2: Convert JSON to Arrow RecordBatch
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("org_id", DataType::UInt64, false),
        Field::new("user_id", DataType::Utf8, false),
        Field::new("event_type", DataType::Utf8, false),
        Field::new("amount", DataType::Float64, true),
        Field::new("country", DataType::Utf8, false),
    ]));

    // Extract data into columnar format
    let timestamps: Vec<i64> = json_data
        .iter()
        .map(|e| e.timestamp.timestamp_millis())
        .collect();
    let org_ids: Vec<u64> = json_data.iter().map(|e| e.org_id).collect();
    let user_ids: Vec<String> = json_data.iter().map(|e| e.user_id.clone()).collect();
    let event_types: Vec<String> = json_data.iter().map(|e| e.event_type.clone()).collect();
    let amounts: Vec<Option<f64>> = json_data.iter().map(|e| e.amount).collect();
    let countries: Vec<String> = json_data.iter().map(|e| e.country.clone()).collect();

    // Create Arrow arrays
    let timestamp_array = Arc::new(TimestampMillisecondArray::from(timestamps)) as ArrayRef;
    let org_id_array = Arc::new(UInt64Array::from(org_ids)) as ArrayRef;
    let user_id_array = Arc::new(StringArray::from(user_ids)) as ArrayRef;
    let event_type_array = Arc::new(StringArray::from(event_types)) as ArrayRef;
    let amount_array = Arc::new(arrow::array::Float64Array::from(amounts)) as ArrayRef;
    let country_array = Arc::new(StringArray::from(countries)) as ArrayRef;

    // Create RecordBatch
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            timestamp_array,
            org_id_array,
            user_id_array,
            event_type_array,
            amount_array,
            country_array,
        ],
    )?;

    println!("=== RecordBatch Created ===");
    println!("Schema: {:?}", batch.schema());
    println!("Rows: {}", batch.num_rows());
    println!();

    // Step 3: Push through DataFusion with partition SQL
    let ctx = SessionContext::new();

    // Create an in-memory table from the RecordBatch
    let mem_table = MemTable::try_new(schema.clone(), vec![vec![batch]])?;
    ctx.register_table("events", Arc::new(mem_table))?;

    // Method 1: Using SQL to compute partition columns
    let partition_sql = r#"
        SELECT
            *,
            -- Compute partition columns
            date_trunc('day', timestamp) as partition_date,
            CASE
                WHEN org_id < 200 THEN 'tier1'
                WHEN org_id < 300 THEN 'tier2'
                ELSE 'tier3'
            END as org_tier,
            country as partition_country,
            -- Build a struct (named fields) for partition mapping
            named_struct(
                'year',   EXTRACT(year  FROM timestamp),
                'month',  EXTRACT(month FROM timestamp),
                'day',    EXTRACT(day   FROM timestamp),
                'country', country,
                'org_tier', CASE
                    WHEN org_id < 200 THEN 'tier1'
                    WHEN org_id < 300 THEN 'tier2'
                    ELSE 'tier3'
                END
            ) as partition_struct
        FROM events
        ORDER BY timestamp
    "#;

    let df = ctx.sql(partition_sql).await?;
    let results = df.collect().await?;

    println!("=== Results with Partition Columns (SQL) ===");
    for batch in &results {
        arrow::util::pretty::print_batches(&[batch.clone()])?;
    }
    println!();

    println!("=== Partition Struct Details (Key-Value Pairs) ===");
    print_partition_structs(&results, "partition_struct")?;
    println!();

    // Step 4: Demonstrate per-row streaming approach with MemTable
    println!("=== Per-Row Streaming with MemTable Demo ===");

    // Simulate receiving individual events as they stream in
    let streaming_events = vec![
        Event {
            timestamp: "2025-01-18T10:00:00Z".parse()?,
            org_id: 150,
            user_id: "stream_user1".to_string(),
            event_type: "purchase".to_string(),
            amount: Some(299.99),
            country: "US".to_string(),
        },
        Event {
            timestamp: "2025-01-18T11:00:00Z".parse()?,
            org_id: 250,
            user_id: "stream_user2".to_string(),
            event_type: "login".to_string(),
            amount: None,
            country: "FR".to_string(),
        },
        Event {
            timestamp: "2025-01-18T12:30:00Z".parse()?,
            org_id: 350,
            user_id: "stream_user3".to_string(),
            event_type: "purchase".to_string(),
            amount: Some(199.50),
            country: "DE".to_string(),
        },
    ];

    // Create a RecordBatch for each individual event (simulating streaming)
    let mut individual_batches = Vec::new();

    println!("Processing events as they arrive (one RecordBatch per event):");
    for (i, event) in streaming_events.iter().enumerate() {
        println!("  Processing event {}: {:?}", i + 1, event);

        // Create a single-row RecordBatch for this event
        let single_batch = create_single_event_batch(event, &schema)?;
        individual_batches.push(single_batch);
    }

    println!(
        "Created {} individual RecordBatches",
        individual_batches.len()
    );

    // Create a MemTable where each RecordBatch becomes a partition
    // MemTable expects Vec<Vec<RecordBatch>> where outer Vec = partitions, inner Vec = batches per partition
    let partitions: Vec<Vec<RecordBatch>> = individual_batches
        .into_iter()
        .map(|batch| vec![batch]) // Each batch becomes its own partition
        .collect();

    let streaming_mem_table = MemTable::try_new(schema.clone(), partitions)?;
    ctx.register_table("streaming_events", Arc::new(streaming_mem_table))?;

    // Query the streaming data using the same partition logic
    let streaming_sql = r#"
        SELECT
            user_id,
            timestamp,
            org_id,
            country,
            named_struct(
                'year',   EXTRACT(year  FROM timestamp),
                'month',  EXTRACT(month FROM timestamp),
                'day',    EXTRACT(day   FROM timestamp),
                'country', country,
                'org_tier', CASE
                    WHEN org_id < 200 THEN 'tier1'
                    WHEN org_id < 300 THEN 'tier2'
                    ELSE 'tier3'
                END
            ) as partition_struct
        FROM streaming_events
        ORDER BY timestamp
    "#;

    let df_streaming = ctx.sql(streaming_sql).await?;
    let streaming_results = df_streaming.collect().await?;

    println!("=== Streaming Results (from MemTable with per-event partitions) ===");
    for batch in &streaming_results {
        arrow::util::pretty::print_batches(&[batch.clone()])?;
    }
    println!();

    println!("=== Streaming Partition Struct Details ===");
    print_partition_structs(&streaming_results, "partition_struct")?;

    Ok(())
}

// Helper function to create RecordBatch from events (kept for potential future use)
#[allow(dead_code)]
fn create_record_batch(
    events: &[Event],
    schema: &Arc<Schema>,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let timestamps: Vec<i64> = events
        .iter()
        .map(|e| e.timestamp.timestamp_millis())
        .collect();
    let org_ids: Vec<u64> = events.iter().map(|e| e.org_id).collect();
    let user_ids: Vec<String> = events.iter().map(|e| e.user_id.clone()).collect();
    let event_types: Vec<String> = events.iter().map(|e| e.event_type.clone()).collect();
    let amounts: Vec<Option<f64>> = events.iter().map(|e| e.amount).collect();
    let countries: Vec<String> = events.iter().map(|e| e.country.clone()).collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(TimestampMillisecondArray::from(timestamps)) as ArrayRef,
            Arc::new(UInt64Array::from(org_ids)) as ArrayRef,
            Arc::new(StringArray::from(user_ids)) as ArrayRef,
            Arc::new(StringArray::from(event_types)) as ArrayRef,
            Arc::new(arrow::array::Float64Array::from(amounts)) as ArrayRef,
            Arc::new(StringArray::from(countries)) as ArrayRef,
        ],
    )?;

    Ok(batch)
}

// Helper function to create RecordBatch from a single event
fn create_single_event_batch(
    event: &Event,
    schema: &Arc<Schema>,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(TimestampMillisecondArray::from(vec![
                event.timestamp.timestamp_millis(),
            ])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![event.org_id])) as ArrayRef,
            Arc::new(StringArray::from(vec![event.user_id.clone()])) as ArrayRef,
            Arc::new(StringArray::from(vec![event.event_type.clone()])) as ArrayRef,
            Arc::new(arrow::array::Float64Array::from(vec![event.amount])) as ArrayRef,
            Arc::new(StringArray::from(vec![event.country.clone()])) as ArrayRef,
        ],
    )?;

    Ok(batch)
}

fn print_partition_structs(
    batches: &[RecordBatch],
    struct_col_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for batch in batches {
        let struct_col_index = batch.schema().index_of(struct_col_name)?;
        let struct_col = batch.column(struct_col_index);
        let struct_array = struct_col
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or("partition_struct column is not a StructArray")?;

        // Get field metadata
        let fields = match struct_array.data_type() {
            DataType::Struct(fields) => fields.clone(),
            other => return Err(format!("expected Struct type, got {:?}", other).into()),
        };

        for row in 0..batch.num_rows() {
            // Get the user_id for better context (if available)
            let user_info = if let Ok(user_idx) = batch.schema().index_of("user_id") {
                let user_col = batch.column(user_idx);
                if let Some(user_array) = user_col.as_any().downcast_ref::<StringArray>() {
                    format!(" (user: {})", user_array.value(row))
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            println!("Row {}{}:", row, user_info);

            // Iterate through each field in the struct
            for (field_idx, field) in fields.iter().enumerate() {
                let child_array = struct_array.column(field_idx);

                // Get value based on the field's data type
                let value = match field.data_type() {
                    DataType::Int32 => {
                        if let Some(arr) = child_array
                            .as_any()
                            .downcast_ref::<arrow::array::Int32Array>()
                        {
                            if arr.is_null(row) {
                                "null".to_string()
                            } else {
                                format!("{:?}", arr.value(row))
                            }
                        } else {
                            "type_mismatch".to_string()
                        }
                    }
                    DataType::Int64 => {
                        if let Some(arr) = child_array
                            .as_any()
                            .downcast_ref::<arrow::array::Int64Array>()
                        {
                            if arr.is_null(row) {
                                "null".to_string()
                            } else {
                                format!("{:?}", arr.value(row))
                            }
                        } else {
                            "type_mismatch".to_string()
                        }
                    }
                    DataType::UInt32 => {
                        if let Some(arr) = child_array
                            .as_any()
                            .downcast_ref::<arrow::array::UInt32Array>()
                        {
                            if arr.is_null(row) {
                                "null".to_string()
                            } else {
                                format!("{:?}", arr.value(row))
                            }
                        } else {
                            "type_mismatch".to_string()
                        }
                    }
                    DataType::Utf8 => {
                        if let Some(arr) = child_array.as_any().downcast_ref::<StringArray>() {
                            if arr.is_null(row) {
                                "null".to_string()
                            } else {
                                format!("{:?}", arr.value(row))
                            }
                        } else {
                            "type_mismatch".to_string()
                        }
                    }
                    DataType::Float64 => {
                        if let Some(arr) = child_array
                            .as_any()
                            .downcast_ref::<arrow::array::Float64Array>()
                        {
                            if arr.is_null(row) {
                                "null".to_string()
                            } else {
                                format!("{:?}", arr.value(row))
                            }
                        } else {
                            "type_mismatch".to_string()
                        }
                    }
                    _ => {
                        // Fallback to array_value_to_string for other types
                        array_value_to_string(child_array.as_ref(), row)
                            .unwrap_or_else(|_| "error".to_string())
                    }
                };

                println!(
                    "\t{}: {} (type: {:?})",
                    field.name(),
                    value,
                    field.data_type()
                );
            }
        }
        // A more compact way:

        for row in 0..batch.num_rows() {
            // Get the user_id for better context (if available)
            let user_info = if let Ok(user_idx) = batch.schema().index_of("user_id") {
                let user_col = batch.column(user_idx);
                if let Some(user_array) = user_col.as_any().downcast_ref::<StringArray>() {
                    format!(" (user: {})", user_array.value(row))
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            println!("Row {}{}:", row, user_info);
            for (field_idx, field) in fields.iter().enumerate() {
                let child = struct_array.column(field_idx);
                let value = array_value_to_string(child.as_ref(), row)?;
                println!("\t{}: {}", field.name(), value);
            }
        }
    }
    Ok(())
}
