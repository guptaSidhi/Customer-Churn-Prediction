import sys
import os
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.utils import load_object
from src.config import *

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join(CURR_DIR,'artifacts','model.pkl')

            print("Before Loading")
            model = load_object(file_path=model_path)
            
            print("After Loading")
            scaler = StandardScaler()
            data_scaled = scaler.transform(features)
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
# This will help us in connecting the backend files and predicting purpose
class CustomData:
    def __init__(self, 
                 MonthlyCharges: float,
                 TotalCharges: float,
                 gender_Male: bool,
                 SeniorCitizen_Yes: bool,
                 Partner_Yes: bool,
                 Dependents_Yes: bool,
                 PhoneService_Yes: bool,
                 MultipleLines_No_phone_service: bool,
                 MultipleLines_Yes: bool,
                 InternetService_Fiber_optic: bool,
                 InternetService_No: bool,
                 OnlineSecurity_No_internet_service: bool,
                 OnlineSecurity_Yes: bool,
                 OnlineBackup_No_internet_service: bool,
                 OnlineBackup_Yes: bool,
                 DeviceProtection_No_internet_service: bool,
                 DeviceProtection_Yes: bool,
                 TechSupport_No_internet_service: bool,
                 TechSupport_Yes: bool,
                 StreamingTV_No_internet_service: bool,
                 StreamingTV_Yes: bool,
                 StreamingMovies_No_internet_service: bool,
                 StreamingMovies_Yes: bool,
                 Contract_One_year: bool,
                 Contract_Two_year: bool,
                 PaperlessBilling_Yes: bool,
                 PaymentMethod_Credit_card_automatic: bool,
                 PaymentMethod_Electronic_check: bool,
                 PaymentMethod_Mailed_check: bool,
                 tenure_group_13_24: bool,
                 tenure_group_25_36: bool,
                 tenure_group_37_48: bool,
                 tenure_group_49_60: bool,
                 tenure_group_61_72: bool):
        # Assign all fields
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges
        self.gender_Male = gender_Male
        self.SeniorCitizen_Yes = SeniorCitizen_Yes
        self.Partner_Yes = Partner_Yes
        self.Dependents_Yes = Dependents_Yes
        self.PhoneService_Yes = PhoneService_Yes
        self.MultipleLines_No_phone_service = MultipleLines_No_phone_service
        self.MultipleLines_Yes = MultipleLines_Yes
        self.InternetService_Fiber_optic = InternetService_Fiber_optic
        self.InternetService_No = InternetService_No
        self.OnlineSecurity_No_internet_service = OnlineSecurity_No_internet_service
        self.OnlineSecurity_Yes = OnlineSecurity_Yes
        self.OnlineBackup_No_internet_service = OnlineBackup_No_internet_service
        self.OnlineBackup_Yes = OnlineBackup_Yes
        self.DeviceProtection_No_internet_service = DeviceProtection_No_internet_service
        self.DeviceProtection_Yes = DeviceProtection_Yes
        self.TechSupport_No_internet_service = TechSupport_No_internet_service
        self.TechSupport_Yes = TechSupport_Yes
        self.StreamingTV_No_internet_service = StreamingTV_No_internet_service
        self.StreamingTV_Yes = StreamingTV_Yes
        self.StreamingMovies_No_internet_service = StreamingMovies_No_internet_service
        self.StreamingMovies_Yes = StreamingMovies_Yes
        self.Contract_One_year = Contract_One_year
        self.Contract_Two_year = Contract_Two_year
        self.PaperlessBilling_Yes = PaperlessBilling_Yes
        self.PaymentMethod_Credit_card_automatic = PaymentMethod_Credit_card_automatic
        self.PaymentMethod_Electronic_check = PaymentMethod_Electronic_check
        self.PaymentMethod_Mailed_check = PaymentMethod_Mailed_check
        self.tenure_group_13_24 = tenure_group_13_24
        self.tenure_group_25_36 = tenure_group_25_36
        self.tenure_group_37_48 = tenure_group_37_48
        self.tenure_group_49_60 = tenure_group_49_60
        self.tenure_group_61_72 = tenure_group_61_72

    def get_data_as_dataframe(self):
        try:
            # Create a dictionary with the data
            custom_data_input_dict = {
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
                "gender_Male": [self.gender_Male],
                "SeniorCitizen_Yes": [self.SeniorCitizen_Yes],
                "Partner_Yes": [self.Partner_Yes],
                "Dependents_Yes": [self.Dependents_Yes],
                "PhoneService_Yes": [self.PhoneService_Yes],
                "MultipleLines_No phone service": [self.MultipleLines_No_phone_service],
                "MultipleLines_Yes": [self.MultipleLines_Yes],
                "InternetService_Fiber optic": [self.InternetService_Fiber_optic],
                "InternetService_No": [self.InternetService_No],
                "OnlineSecurity_No internet service": [self.OnlineSecurity_No_internet_service],
                "OnlineSecurity_Yes": [self.OnlineSecurity_Yes],
                "OnlineBackup_No internet service": [self.OnlineBackup_No_internet_service],
                "OnlineBackup_Yes": [self.OnlineBackup_Yes],
                "DeviceProtection_No internet service": [self.DeviceProtection_No_internet_service],
                "DeviceProtection_Yes": [self.DeviceProtection_Yes],
                "TechSupport_No internet service": [self.TechSupport_No_internet_service],
                "TechSupport_Yes": [self.TechSupport_Yes],
                "StreamingTV_No internet service": [self.StreamingTV_No_internet_service],
                "StreamingTV_Yes": [self.StreamingTV_Yes],
                "StreamingMovies_No internet service": [self.StreamingMovies_No_internet_service],
                "StreamingMovies_Yes": [self.StreamingMovies_Yes],
                "Contract_One year": [self.Contract_One_year],
                "Contract_Two year": [self.Contract_Two_year],
                "PaperlessBilling_Yes": [self.PaperlessBilling_Yes],
                "PaymentMethod_Credit card (automatic)": [self.PaymentMethod_Credit_card_automatic],
                "PaymentMethod_Electronic check": [self.PaymentMethod_Electronic_check],
                "PaymentMethod_Mailed check": [self.PaymentMethod_Mailed_check],
                "tenure_group_13 - 24": [self.tenure_group_13_24],
                "tenure_group_25 - 36": [self.tenure_group_25_36],
                "tenure_group_37 - 48": [self.tenure_group_37_48],
                "tenure_group_49 - 60": [self.tenure_group_49_60],
                "tenure_group_61 - 72": [self.tenure_group_61_72],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
