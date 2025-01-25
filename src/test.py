import os
import pandas as pd
def verify_file(file_path):
    if not os.path.exists(file_path):
        logger.error('File does not exist: %s', file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

verify_file('./data/interim/train_processed.csv')
verify_file('./data/interim/test_processed.csv')
data = pd.read_csv('./data/interim/train_processed.csv')
print(data.head())