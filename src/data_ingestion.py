import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import logging
import requests
import yaml

log_dir = "logs"
os.makedirs(log_dir,exist_ok =True) # checks whether the directory exists, if yes it doesnt overwrite

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)-> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.debug('parameters savely retrieved from yaml')
            return params
    except Exception as e:
        logger.error('unexpected error has occured as %s',e)
        raise

def load_data(url):
    try:
        df = pd.read_csv(url)
        logger.debug(f'data loaded from {url}')
    except pd.errors.ParserError as e:
            logger.debug(f'failed to parse data from the csv file - {e}')
            raise
    except Exception as e:
        logger.debug(f'unexpected error has happened - {e}')
        raise

def preprocess_data(df):
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data,test_data,data_path):
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok = True)
        train_data.to_csv(os.path.join(raw_data_path,'train_data.csv'),index =False)
        test_data.to_csv(os.path.join(raw_data_path,'test_data.csv'),index=False)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path = 'params.yaml')
        test_size = params['data_ingestion']['test_size']

        url = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        response = requests.get(url)
        if response.status_code == 200:
            with open('spam.csv', 'wb') as f:
                f.write(response.content)
            df = pd.read_csv('spam.csv')
        preprocessed_data = preprocess_data(df)
        train_data, test_data = train_test_split(preprocessed_data,test_size = test_size,random_state = 42)
        save_data(train_data,test_data,'./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()