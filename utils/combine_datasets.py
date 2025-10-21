"""
Script to clean and combine all OBD-II dataset CSV files into a single CSV file for Watson X.
This script:
1. Removes NaN rows from all OBD-II dataset files and saves them to the trimmed folder
2. Combines all trimmed files into one consolidated data asset with a source file identifier column
"""
import pandas as pd
import glob
import os
from pathlib import Path


def clean_and_save_datasets(input_dir, output_dir):
    """
    Clean all CSV files by removing NaN rows and save to trimmed directory.
    
    Args:
        input_dir: Directory containing the original CSV files
        output_dir: Directory to save the cleaned CSV files
    
    Returns:
        Number of files processed
    """
    # Get all CSV files in the input directory
    csv_pattern = os.path.join(input_dir, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return 0
    
    print(f"\n{'='*60}")
    print(f"STEP 1: Cleaning {len(csv_files)} CSV files (removing NaN rows)")
    print(f"{'='*60}\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    files_processed = 0
    total_rows_before = 0
    total_rows_after = 0
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        filename = os.path.basename(csv_file)
        print(f"[{i}/{len(csv_files)}] Processing: {filename}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            rows_before = len(df)
            total_rows_before += rows_before
            
            # Remove rows with any NaN values
            df_cleaned = df.dropna()
            rows_after = len(df_cleaned)
            total_rows_after += rows_after
            rows_removed = rows_before - rows_after
            
            # Save cleaned file to trimmed directory
            output_path = os.path.join(output_dir, filename)
            df_cleaned.to_csv(output_path, index=False)
            
            print(f"  ✓ Rows before: {rows_before:,} | After: {rows_after:,} | Removed: {rows_removed:,} ({rows_removed/rows_before*100:.1f}%)")
            files_processed += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Cleaning Summary:")
    print(f"  - Files processed: {files_processed}/{len(csv_files)}")
    print(f"  - Total rows before: {total_rows_before:,}")
    print(f"  - Total rows after: {total_rows_after:,}")
    print(f"  - Total rows removed: {total_rows_before - total_rows_after:,} ({(total_rows_before - total_rows_after)/total_rows_before*100:.1f}%)")
    print(f"  - Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    return files_processed


def combine_obd_datasets(input_dir, output_file):
    """
    Combine all CSV files from the OBD-II dataset into a single CSV file.
    
    Args:
        input_dir: Directory containing the CSV files
        output_file: Path to the output combined CSV file
    """
    # Get all CSV files in the input directory
    csv_pattern = os.path.join(input_dir, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return None
    
    print(f"\n{'='*60}")
    print(f"STEP 2: Combining {len(csv_files)} CSV files into one dataset")
    print(f"{'='*60}\n")
    
    # List to store all dataframes
    all_dfs = []
    
    # Read and process each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Processing: {os.path.basename(csv_file)}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Preserve the original row index from the file
            df['original_row_id'] = df.index
            
            # Add a column to identify the source file
            df['source_file'] = os.path.basename(csv_file)
            
            # Create a unique composite ID: filename_rowid
            filename_base = os.path.basename(csv_file).replace('.csv', '')
            df['unique_id'] = filename_base + '_row_' + df['original_row_id'].astype(str)
            
            # Extract metadata from filename (e.g., date, route, traffic condition)
            parts = filename_base.split('_')
            
            if len(parts) >= 2:
                df['date'] = parts[0]  # Date (e.g., 2017-07-05)
                df['vehicle'] = '_'.join(parts[1:3]) if len(parts) >= 3 else 'Unknown'  # Vehicle info
                
                # Extract route and traffic info if available
                if len(parts) >= 4:
                    df['route_from'] = parts[3] if len(parts) > 3 else 'Unknown'
                    df['route_to'] = parts[4] if len(parts) > 4 else 'Unknown'
                    df['traffic_condition'] = '_'.join(parts[5:]) if len(parts) > 5 else 'Unknown'
                else:
                    df['route_from'] = 'Unknown'
                    df['route_to'] = 'Unknown'
                    df['traffic_condition'] = 'Unknown'
                
                # Combine date and Time columns into a proper datetime timestamp
                # This is crucial for time series analysis
                if 'Time' in df.columns:
                    # Convert to datetime object first
                    datetime_obj = pd.to_datetime(df['date'] + ' ' + df['Time'].astype(str))
                    # Convert to UNIX timestamp (seconds since epoch)
                    df['timestamp'] = datetime_obj.astype('int64') // 10**9
                    # Keep the original Time column for reference if needed
                    df.rename(columns={'Time': 'Time_original'}, inplace=True)
            
            all_dfs.append(df)
            print(f"  - Rows: {len(df)}, Columns: {len(df.columns)}")
            
        except Exception as e:
            print(f"  - Error processing {csv_file}: {str(e)}")
            continue
    
    if not all_dfs:
        print("No data to combine!")
        return None
    
    # Combine all dataframes
    print("\n" + "="*60)
    print("Merging all dataframes into single dataset...")
    print("="*60)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Reorder columns to put timestamp first (if it exists)
    if 'timestamp' in combined_df.columns:
        cols = combined_df.columns.tolist()
        cols.remove('timestamp')
        combined_df = combined_df[['timestamp'] + cols]
    
    print(f"\nCombined dataset:")
    print(f"  - Total rows: {len(combined_df)}")
    print(f"  - Total columns: {len(combined_df.columns)}")
    print(f"  - Columns: {list(combined_df.columns)}")
    print(f"\nUnique source files: {combined_df['source_file'].nunique()}")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    # Save to output file
    print(f"\nSaving combined dataset to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    print("Done!")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Output file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024**2):.2f} MB")
    print(f"Total rows: {len(combined_df):,}")
    print(f"Total columns: {len(combined_df.columns)}")
    
    return combined_df


def main():
    """Main function to run the dataset cleaning and combination."""
    # Define paths
    project_root = Path(__file__).parent.parent
    
    # Paths for cleaning
    full_input_dir = project_root / "data" / "dataset" / "OBD-II-Dataset"
    trimmed_output_dir = project_root / "data" / "dataset" / "trimmed"
    
    # Path for final combined file
    final_output_file = project_root / "data" / "dataset" / "combined_obd_dataset_watsonx.csv"
    
    print("="*60)
    print("OBD-II Dataset Processor for Watson X")
    print("="*60)
    print("\nThis script will:")
    print("  1. Clean all OBD-II dataset files (remove NaN rows)")
    print("  2. Save cleaned files to the 'trimmed' directory")
    print("  3. Combine all cleaned files into one CSV for Watson X")
    print("\n" + "="*60)
    
    # Step 1: Clean all datasets
    files_cleaned = clean_and_save_datasets(str(full_input_dir), str(trimmed_output_dir))
    
    if files_cleaned == 0:
        print("No files were cleaned. Exiting.")
        return
    
    # Step 2: Combine all cleaned datasets
    combined_df = combine_obd_datasets(str(trimmed_output_dir), str(final_output_file))
    
    if combined_df is None:
        print("Failed to combine datasets. Exiting.")
        return
    
    print("\n" + "="*60)
    print("✓ ALL DONE!")
    print("="*60)
    print(f"\nYour Watson X data asset is ready:")
    print(f"  📄 File: {final_output_file}")
    print(f"  📊 Rows: {len(combined_df):,}")
    print(f"  📋 Columns: {len(combined_df.columns)}")
    print(f"  💾 Size: {os.path.getsize(final_output_file) / (1024**2):.2f} MB")
    print(f"\n  🔑 ID Columns:")
    print(f"     - unique_id: Composite ID (filename + row number)")
    print(f"     - original_row_id: Original row index from source file")
    print(f"     - source_file: Source filename")
    print("\nYou can now upload this CSV file to Watson X as a single data asset.")
    print("="*60)


if __name__ == "__main__":
    main()

