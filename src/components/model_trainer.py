import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object ,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {

                "Random Forest": RandomForestClassifier(),

                "Decision Tree": DecisionTreeClassifier(),

                "Gradient Boosting": GradientBoostingClassifier(),

                "Logistic Regression": LogisticRegression(),

                "KNN": KNeighborsClassifier(),

                "XGBoost": XGBClassifier(
                    eval_metric='logloss'
                ),

                "CatBoost": CatBoostClassifier(
                    verbose=False
                ),

                "AdaBoost": AdaBoostClassifier()
            }

            params = {

                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [5,10,None]
                },

                "Random Forest": {
                    "n_estimators": [50,100,200],
                    "max_depth": [5,10,None]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01,0.1],
                    "n_estimators": [100,200]
                },

                "Logistic Regression": {
                    "C": [0.01,0.1,1,10]
                },

                "KNN": {
                    "n_neighbors": [3,5,7,9]
                },

                "XGBoost": {
                    "n_estimators": [100,200],
                    "learning_rate": [0.01,0.1],
                    "max_depth": [3,6]
                },

                "CatBoost": {
                    "iterations": [100,200],
                    "learning_rate": [0.01,0.1]
                },

                "AdaBoost": {
                    "n_estimators": [50,100,200],
                    "learning_rate": [0.01,0.1]
                }
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,params=params)
            
            ## To get best model score from dict
            best_model_score = max((model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # if best_model_score<0.6:
            #     raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predicted
            )

            return accuracy
            
            
        except Exception as e:
            raise CustomException(e,sys)