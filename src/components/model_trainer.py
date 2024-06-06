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
                "XGBRegressor": XGBClassifier(),
                "SVC":SVC(),
                "AdaBoost": AdaBoostClassifier(),
            }

            # params={
            #     "Decision Tree": {
            #         'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            #         # 'splitter':['best','random'],
            #         # 'max_features':['sqrt','log2'],
            #     },
            #     "Random Forest":{
            #         # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
            #         # 'max_features':['sqrt','log2',None],
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "Gradient Boosting":{
            #         # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
            #         'learning_rate':[.1,.01,.05,.001],
            #         'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
            #         # 'criterion':['squared_error', 'friedman_mse'],
            #         # 'max_features':['auto','sqrt','log2'],
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "Linear Regression":{},
            #     "XGBRegressor":{
            #         'learning_rate':[.1,.01,.05,.001],
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "CatBoosting Regressor":{
            #         'depth': [6,8,10],
            #         'learning_rate': [0.01, 0.05, 0.1],
            #         'iterations': [30, 50, 100]
            #     },
            #     "AdaBoost Regressor":{
            #         'learning_rate':[.1,.01,0.5,.001],
            #         # 'loss':['linear','square','exponential'],
            #         'n_estimators': [8,16,32,64,128,256]
            #     }
                
            # }

            model_report,roc_report = evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test,y_test = y_test,models=models)

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