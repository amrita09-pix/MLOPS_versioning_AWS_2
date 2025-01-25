import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import logging

log_dir = 'logs'
og_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path:str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s',file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error has happened %s',e)
        raise

def load_data(file_path:str)->None:
    try:
        with open('your_file.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        logger.error('Data loaded from %s',file_path)
        return loaded_model
    except Exception as e:
        logger.error('Unexpected error has happened %s',e)
        raise
 
def evaluate_model(clf,x_test:np.array, y_test:np.array)-> None:
    try:
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:,-1]

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,y_pred_proba)
        logger.debug('evaluation metrics calculated')
        metrics_dict = {'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'roc_auc_score':roc_auc
        }
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics_dict:dict,file_path)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok = True)

        with open(file_path, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
            logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    model =load_model('/Users/amritamandal/MLOPS_new_2/MLOPS_versioning_AWS_2/models/model.pkl')
    test_data = pd.read_csv('/Users/amritamandal/MLOPS_new_2/MLOPS_versioning_AWS_2/data/processed/test_tfidf.csv')
    x_test = test_data.iloc[:,:-1].values
    y_test = test_data.iloc[:,-1].values
    metrics = evaluate_model(model,x_test,y_test)
    save_metrics(metrics, 'report/metrics.json')

if __name__=='__main__':
    main()









