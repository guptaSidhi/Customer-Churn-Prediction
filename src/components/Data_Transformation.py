import os 
import sys 
import pandas as pd

from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.config import *

@dataclass
class DataTransformationConfig:
    filepath : str = os.path.join(CURR_DIR,'artifacts','correct_raw.csv')

class DataTransformation:
    def __init__(self):
        self.transformationconfig = DataTransformationConfig()

    def initiate_data_transformation(self):
        try:
            # Reading new csv file
            df = pd.read_csv(self.transformationconfig.filepath)
            logging.info("Reading the Correct File Path")

            # Making record of columns
            numerical_columns = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_columns = [feature for feature in df.columns if df[feature].dtype == 'O']

            X = df.drop(columns={'Churn'},axis=1)
            y = df['Churn']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logging.info("Standard Scaling Done!")

            sm = SMOTEENN(sampling_strategy='minority',random_state=42)
            X_resampled2, y_resampled2 = sm.fit_resample(X_scaled,y)
            logging.info("Resampling Done!")

            column_names = list(df.columns)

            # Find the index of the 'Churn' column
            target_column = 'Churn'
            target_column_index = column_names.index(target_column)

            # Separate feature columns and target column
            feature_columns = column_names[:target_column_index] + column_names[target_column_index+1:]

            # Convert X_resampled and y_resampled to DataFrames with appropriate column names
            X_resampled_df = pd.DataFrame(X_resampled2, columns=feature_columns)
            y_resampled_df = pd.DataFrame(y_resampled2, columns=[target_column])

            # Concatenate the features and target DataFrames along the columns
            # Insert 'Churn' column back to its original position
            resampled_df = pd.concat([X_resampled_df, y_resampled_df], axis=1)

            # Reorder columns to match the original DataFrame column order
            resampled_df = resampled_df[column_names]
            filepath_resampled = os.path.join(CURR_DIR,'artifacts','resampled_raw_data.csv')
            resampled_df.to_csv(filepath_resampled,index=False)
            logging.info("Converting Resampling Data back to Dataframe")

        except Exception as e:
            raise CustomException(e,sys)
        
    