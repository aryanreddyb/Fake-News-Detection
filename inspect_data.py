import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "data")
files = ["multimodal_train.tsv", "multimodal_validate.tsv", "multimodal_test_public.tsv"]

def inspect_data():
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        
        print(f"--- Inspecting {f} ---")
        df = pd.read_csv(path, sep='\t')
        print(f"Dimensions: {df.shape}")
        print("\nColumns and Data Types:")
        print(df.dtypes)
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nDuplicates:", df.duplicated().sum())
        
        if '2_way_label' in df.columns:
            print("\n2_way_label distribution:")
            print(df['2_way_label'].value_counts(dropna=False))
            
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    inspect_data()
