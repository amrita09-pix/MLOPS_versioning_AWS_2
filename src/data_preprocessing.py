import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

encoder = LabelEncoder()
stemmer = PorterStemmer()

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_dir = 'logs'
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize text
    
    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric tokens
            y.append(i)
    
    # Remove stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    # Apply stemming using PorterStemmer
    text = y[:]
    y.clear()
    for i in text:
        y.append(stemmer.stem(i))
    
    return " ".join(y)

def preprocess(df,text_column,target_column):
    df[target_column] = encoder.fit_transform(df[target_column])
    df = df.drop_duplicates(keep = 'first')
    df['transformed_text'] = df[text_column].apply(transform_text)
    return df

def main():
    train_data = pd.read_csv('./data/raw/train_data.csv')
    test_data = pd.read_csv('./data/raw/test_data.csv')
    train_data_preprocessed = preprocess(train_data,'text','target')
    test_data_preprocessed = preprocess(test_data,'text','target')
    data_path = os.path.join("./data", "interim")
    os.makedirs(data_path, exist_ok=True)
    train_data_preprocessed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
    test_data_preprocessed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
    logger.debug('Processed data saved to %s', data_path)

if __name__ =='__main__':
    main()
    
