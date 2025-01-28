import os
import numpy as np
import pandas as pd
import pickle
import logging 
from sklearn.ensemble import RandomForestClassifier
import yaml

log_dir = 'logs'

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
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

def load_data(file_path:str):
    try:
        df = pd.read_csv(file_path)
        logger.debug('data loaded from %s with shape %s',file_path,df.shape)
        return df
    except Exception as e:
        logger.error('unexpected error has occured -%s ' ,e)
        raise 

def train_model(x_train,y_train,params):
    try:
        clf = RandomForestClassifier(n_estimators = params['n_estimators'],random_state = params['random_state'])
        logger.debug('initialising random forest with parameters: %s',params)
        clf.fit(x_train,y_train)
        logger.debug('model training completed')
        return clf
    
    except ValueError as e:
        logger.error(f'Value error during model training -{e}')
        raise
    except Exception as e:
        logger.error(f'unexpected error has occured - {e}')
        raise

def save_model(model, file_path:str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok = True)
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('model saved to %s',file_path)
    except FileNotFoundError as e:
        logger.error ('file path not found %s',e)
        raise
    except Exception as e:
        logger.error(f'unexpected error has occured - {e}')
        raise

def main():
    try:
        params = load_params(params_path = 'params.yaml')
        n_estimators = params['model_training']['n_estimators']
        random_state = params['model_training']['random_state']
        params = {'n_estimators':n_estimators ,'random_state':random_state}
        train_data = load_data('/Users/amritamandal/MLOPS_new_2/MLOPS_versioning_AWS_2/data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(x_train,y_train,params)
        model_save_path = 'models/model.pkl'
        save_model(clf,model_save_path)
    except Exception as e:
        logger.error('failed to complete the model building process %s',e)
        print(f'error :{e}')

if __name__=='__main__':
    main()