use parquet::errors::Result;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;
use std::path::Path;

fn inspect_parquet_layout(file_path: &str) -> Result<()> {
    let path = Path::new(file_path);
    let mut file = File::open(&path)?;
    let file_size = file.metadata()?.len();

    // Read the actual footer length from the last 8 bytes of the file
    // Last 8 bytes: [footer_length: 4 bytes][magic "PAR1": 4 bytes]
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::End(-8))?;
    let mut footer_info = [0u8; 8];
    file.read_exact(&mut footer_info)?;

    // Extract footer length (little-endian)
    let footer_len = u32::from_le_bytes([
        footer_info[0],
        footer_info[1],
        footer_info[2],
        footer_info[3],
    ]) as u64;
    let magic = &footer_info[4..8];

    // Verify magic number
    if magic != b"PAR1" {
        return Err(parquet::errors::ParquetError::General(
            "Invalid Parquet file: missing PAR1 magic".to_string(),
        ));
    }

    let footer_offset = file_size - footer_len - 8;

    // Reset file position and create reader
    file.seek(SeekFrom::Start(0))?;
    let reader = SerializedFileReader::new(file)?;
    let parquet_metadata = reader.metadata();

    println!("--- File Layout ---");
    println!("File Size: {} bytes", file_size);
    println!("Footer Offset: {}", footer_offset);
    println!("Footer Length: {} bytes", footer_len);
    println!(
        "Magic Number: {:?}",
        std::str::from_utf8(magic).unwrap_or("Invalid")
    );
    println!("-------------------\n");
    println!("The following information is from the footer:");

    // 2. Row Group Metadata Locations
    for (i, row_group_meta) in parquet_metadata.row_groups().iter().enumerate() {
        if let Some(file_offset) = row_group_meta.file_offset() {
            println!("\n--- Row Group {} ---", i);
            println!("  Offset: {}", file_offset);
            println!("  Total Byte Size: {}", row_group_meta.total_byte_size());
            println!("  Compressed Size: {}", row_group_meta.compressed_size());
            println!("  Number of Rows: {}", row_group_meta.num_rows());

            // 3. Column Chunk Metadata Locations
            for (j, col_chunk_meta) in row_group_meta.columns().iter().enumerate() {
                println!("\n  --- Column Chunk {} in Row Group {} ---", j, i);
                println!("    Column Path: {}", col_chunk_meta.column_path());
                println!(
                    "    Data Page Offset: {}",
                    col_chunk_meta.data_page_offset()
                );
                if let Some(dict_offset) = col_chunk_meta.dictionary_page_offset() {
                    println!("    Dictionary Page Offset: {}", dict_offset);
                }
                println!(
                    "    Total Compressed Size: {}",
                    col_chunk_meta.compressed_size()
                );

                if let Some(stats) = col_chunk_meta.statistics() {
                    if stats.min_bytes_opt().is_some() && stats.max_bytes_opt().is_some() {
                        println!("    Statistics available (for predicate pushdown)");
                    }
                }
            }
        } else {
            println!("\n--- Row Group {} ---", i);
            println!("  Offset information not available.");
        }
    }

    Ok(())
}

fn main() {
    // Create a dummy parquet file for demonstration
    // In a real scenario, you would have your own parquet file.
    create_dummy_parquet("sample.parquet").unwrap();

    if let Err(e) = inspect_parquet_layout("sample.parquet") {
        eprintln!("Error inspecting parquet file: {}", e);
    }
}

// Helper function to create a dummy parquet file
use arrow::record_batch::RecordBatch;
use arrow_array::{Int32Array, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use std::sync::Arc;

fn create_dummy_parquet(file_path: &str) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                None,
                Some("d"),
                Some("e"),
            ])),
        ],
    )?;

    let file = File::create(file_path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}
