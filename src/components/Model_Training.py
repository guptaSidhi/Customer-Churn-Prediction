import os 
import sys 

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import *
from src.config import *

from sklearn.metrics import accuracy_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(CURR_DIR,'artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def inititate_model_trainer(self,X_train, X_test, y_train, y_test):
        
        try:

            logging.info("Split Training and Test input Data")
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(), 
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Random Forest": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [50, 100, 150, 200, 250]
                },
                "Gradient Boosting": {
                    'loss': ['log_loss', 'exponential'],
                    # 'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    # 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'max_features': ['sqrt', 'log2'],
                    'n_estimators': [50, 100, 150, 200, 250]
                },
                "Logistic Regression": {
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga', 'lbfgs']
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'n_estimators': [50, 100, 150, 200, 250],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
                },
                "CatBoosting Classifier": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 150]
                },
                "AdaBoost Classifier": {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.1, 0.05, 0.01, 0.001]
                }
            }


            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            rec = recall_score(y_test, predicted)
            return rec
        
        except Exception as e:
            raise CustomException(e,sys)

