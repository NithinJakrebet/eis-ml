import os
import pandas as pd

def load_data(subfolder: str, filename: str, data_dir: str = "../data") -> pd.DataFrame:
    """
    Loads a CSV file from a given subfolder in the data directory.

    Parameters:
        subfolder (str): The name of the subfolder (e.g., "03-06-24").
        filename (str): The name of the CSV file to load (e.g., "A1.csv").
        data_dir (str, optional): The base directory containing data subfolders. Defaults to "../data".

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If an error occurs while reading the file.
    """

    file_path = os.path.join(data_dir, subfolder, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading '{file_path}': {e}")

    # Drop NaN values
    df = df.dropna()

    # Fix column name issue
    if "#NAME?" in df.columns:
        df = df.rename(columns={"#NAME?": "Im(Z)/Ohm"})
        
    # drop the 0th cycle
    df = df.loc[(df['cycle number'] != 0)].copy()


    return df

