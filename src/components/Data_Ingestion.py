import os 
import sys 
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.config import *

from imblearn.combine import SMOTEENN
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.Data_Preprocessing import *
from src.components.Data_Transformation import *
from src.components.Model_Training import *
from src.utils import *

@dataclass
class DataIngestionConfig:
    train_file_path : str = os.path.join(CURR_DIR,'artifacts','train.csv')
    test_file_path : str = os.path.join(CURR_DIR,'artifacts','test.csv')
    raw_file_path : str = os.path.join(CURR_DIR,'artifacts','resampled_raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion Module")
        try:
            df = pd.read_csv(self.ingestion_config.raw_file_path)
            logging.info("Read the Dataset as Dataframe")
            
            logging.info("Train Test Split Initiated")
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)
            train_data.to_csv(self.ingestion_config.train_file_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_file_path,index=False,header=True)
            logging.info('Ingestion of the data is completed')

            # Separate features and target (assuming the target column is 'Churn')
            X_train = train_data.drop(columns=['Churn'])
            y_train = train_data['Churn']
            X_test = test_data.drop(columns=['Churn'])
            y_test = test_data['Churn']
            
            # Convert to NumPy arrays
            X_train_array = X_train.to_numpy()
            y_train_array = y_train.to_numpy()
            X_test_array = X_test.to_numpy()
            y_test_array = y_test.to_numpy()
            
            logging.info("Converted train and test data into arrays")

            return X_train_array, X_test_array, y_train_array, y_test_array
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataPreprocessing()
    obj.initiate_data_preprocessing()
    new_data = DataTransformation()
    new_data.initiate_data_transformation()
    datain = DataIngestion()
    X_train, X_test, y_train, y_test = datain.initiate_data_ingestion()
    model_train = ModelTrainer()
    rec = model_train.inititate_model_trainer(X_train, X_test, y_train, y_test)
    print(rec)