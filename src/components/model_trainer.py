import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training and test input data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(),
                "SVC":SVC(),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                'Decision Tree': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 10, 20],
                    'min_samples_leaf': [1, 5, 10]
                },
                'Random Forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.5],
                    'max_depth': [3, 5, 7]
                },
                'XGBoost': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.5],
                    'max_depth': [3, 5, 7]
                },
                'SVC': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                'AdaBoost':{
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1],
                    'algorithm': ['SAMME', 'SAMME.R']
                }
            }

            model_report,roc_report = evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test,y_test = y_test,models=models,param=params)

            ## To get the best model score from dict
            best_model_score = max(sorted(roc_report.values()))

            ## To get the best model name from dict
            best_model_nmae = list(roc_report.keys())[list(roc_report.values()).index(best_model_score)]

            best_model = models[best_model_nmae]
 
            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info('Best model found on both training and test dataset')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted = best_model.predict(X_test)

            roc_auc = roc_auc_score(y_test,predicted)

            return roc_auc

        except Exception as e:
            raise CustomException(e,sys)