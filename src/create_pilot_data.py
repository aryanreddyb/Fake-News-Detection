import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

def create_pilot():
    print("Creating Pilot Dataset (10,000 rows total)...")
    
    # Load the full processed files
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "validate.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Sample them (e.g., 8000 train, 1000 val, 1000 test)
    # random_state ensures reproducibility
    train_pilot = train_df.sample(n=8000, random_state=42)
    val_pilot = val_df.sample(n=1000, random_state=42)
    test_pilot = test_df.sample(n=1000, random_state=42)
    
    # Save over the original processed files so the rest of our pipeline just works
    train_pilot.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_pilot.to_csv(os.path.join(DATA_DIR, "validate.csv"), index=False)
    test_pilot.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    
    print("Pilot dataset created successfully!")
    print(f"Train: {len(train_pilot)} rows")
    print(f"Val: {len(val_pilot)} rows")
    print(f"Test: {len(test_pilot)} rows")
    print("\n(Note: When you want to run the full 500k dataset later, just re-run python src/preprocessing.py to restore the full files.)")

if __name__ == "__main__":
    create_pilot()
