import pandas as pd
# Load the data to check for NaN and basic structure again
file_path = 'data.csv'
df = pd.read_csv(file_path)
# List columns, dtypes, null counts
basic_info = {
    'columns': df.columns.tolist(),
    'dtypes': df.dtypes.astype(str).to_dict(),
    'null_counts': df.isnull().sum().to_dict(),
    'shape': df.shape,
    'head': df.head().to_dict()
}
basic_info
