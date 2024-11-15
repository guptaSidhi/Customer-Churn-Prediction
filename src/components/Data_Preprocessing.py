import os 
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.config import *
from dataclasses import dataclass

@dataclass
class DataPreprocessingConfig:
    raw_file_path : str = os.path.join(CURR_DIR,'artifacts','raw.csv')

class DataPreprocessing:
    def __init__(self):
        self.preprocessing_config = DataPreprocessingConfig()

    def initiate_data_preprocessing(self):

        try:
            logging.info("Starting to read original dataset")
            df = pd.read_csv(RAW_DATA_PATH)

            os.makedirs(os.path.dirname(self.preprocessing_config.raw_file_path),exist_ok=True)
            df.to_csv(self.preprocessing_config.raw_file_path,index=False,header=False)

            logging.info("Starting the process of Data Preprocessing")

            # Changing the datatypes of some columns
            df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
            df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')

            # Removing Null Values(0.15%)
            df.dropna(inplace=True)

            # Making groups of Tenure 
            labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
            df['tenure_group'] = pd.cut(df.tenure, range(1, 80, 12), right=False, labels=labels)
            df.drop(columns={'customerID','tenure'},inplace=True)

            # Mapping some conditions 
            df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
            df['SeniorCitizen'] = df['SeniorCitizen'].map({0:'No',1:'Yes'})

            # Numerical and Categorical Columns
            numerical_columns = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_columns = [feature for feature in df.columns if df[feature].dtype == 'O']

            logging.info(f"Numerical Columns: {numerical_columns}")
            logging.info(f"Categorical Columns: {categorical_columns}")

            # Converting Columns into One Hot Encoding 
            new_df = pd.get_dummies(df,drop_first=True)

            filepath = os.path.join(CURR_DIR,'artifacts','correct_raw.csv')
            new_df.to_csv(filepath,index=False)
            logging.info("Saving new corrected data into artifacts")

        except Exception as e:
            raise CustomException(e,sys)
        