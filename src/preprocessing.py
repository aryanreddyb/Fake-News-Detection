import pandas as pd
import os

DATA_DIR = r"d:\Capstone\fnd\data"
OUTPUT_DIR = r"d:\Capstone\fnd\data\processed"

# Columns to keep to save RAM
COLUMNS_TO_KEEP = [
    'id', 'author', 'subreddit', 'linked_submission_id', 
    'clean_title', 'created_utc', '2_way_label'
]

def preprocess_dataset(filename, output_filename):
    print(f"Processing {filename}...")
    filepath = os.path.join(DATA_DIR, filename)
    
    # Read only required columns to save memory
    df = pd.read_csv(filepath, sep='\t', usecols=COLUMNS_TO_KEEP)
    
    # Handle missing values in text and metadata
    df['clean_title'] = df['clean_title'].fillna("").astype(str)
    df['author'] = df['author'].fillna("unknown_author").astype(str)
    df['subreddit'] = df['subreddit'].fillna("unknown_subreddit").astype(str)
    
    # For linked_submission_id, missing means it's an original post or not linked
    df['linked_submission_id'] = df['linked_submission_id'].fillna("").astype(str)
    
    # Standardize label column name to 'label'
    df.rename(columns={'2_way_label': 'label'}, inplace=True)
    
    # Drop rows where text is too short after cleaning (optional, but good practice)
    # DistilBERT can handle empty strings, but it's noise for classification
    mask_empty = df['clean_title'].str.strip() == ""
    if mask_empty.sum() > 0:
        print(f"Dropped {mask_empty.sum()} empty text rows from {filename}")
        df = df[~mask_empty]
        
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path} | Shape: {df.shape}")

def main():
    preprocess_dataset("multimodal_train.tsv", "train.csv")
    preprocess_dataset("multimodal_validate.tsv", "validate.csv")
    preprocess_dataset("multimodal_test_public.tsv", "test.csv")
    print("Preprocessing completed.")

if __name__ == "__main__":
    main()
