import os
import glob
import pandas as pd

def combine():
    # Use only the 03-06-24 folder since they have similar conditions.
    folder = "/Users/nithin.jakrebet/Desktop/eis-ml/data/03-06-24"
    print("Absolute folder path:", os.path.abspath(folder))
    print(os.listdir(folder))
    
    meta = {"source": "03-06-24", "C_rate": "High", "temperature": 25}

    all_dfs = []
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder}.")
        return None

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Extract channel info from filename (e.g., "A1.csv" -> "A1")
        channel = os.path.splitext(os.path.basename(csv_file))[0]
        df["channel"] = channel
        df["folder"] = os.path.basename(folder)
        df["source"] = meta.get("source", "")
        df["C_rate"] = meta.get("C_rate", "")
        df["temperature"] = meta.get("temperature", "")
        
        all_dfs.append(df)

    # Concatenate all data into one master DataFrame.
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Write to a file, not a directory.
    combined_df.to_csv("/Users/nithin.jakrebet/Desktop/eis-ml/data/03-06-24/combined_data_03_06.csv", index=False)
    print("Combined dataset shape:", combined_df.shape)
    return combined_df

if __name__ == "__main__":
    combine()
