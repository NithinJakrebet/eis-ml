import os
import pandas as pd

# Define the dataset directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/03-06-24 - Selva, A8 channels"))

def load_data():
    """Loads all CSV files from the specified data directory into a dictionary of DataFrames."""
    
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    data_dict = {}

    for file in data_files:
        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(file_path)
        data_dict[file] = df  # Store each dataset in a dictionary
    
    return data_dict

if __name__ == "__main__":
    data = load_data()
    
    # Print sample output to verify
    for filename, df in data.items(): 
        print(f"\n{filename} - {df.shape}")
        print(df.head())
