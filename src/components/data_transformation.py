import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,RobustScaler,OrdinalEncoder
from category_encoders import OneHotEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    preprocessor_obj_file_path_full_pipeline = os.path.join('artifacts','full_pipeline.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data preprocessing and transformation
        '''
        try:
            numerical_columns = ['number_vmail_messages',
                                    'total_day_minutes', 'total_day_calls', 'total_day_charge',
                                    'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
                                    'total_night_minutes', 'total_night_calls', 'total_night_charge',
                                    'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
                                    'customer_service_calls']
            categorical_columns = ['international_plan','voice_mail_plan']

            state_columns = ['state']

            num_pipe = Pipeline(
                steps=[
                    ('scaler',RobustScaler())
                ]
            )
            logging.info('Numerical columns scaling completed')

            cat_pipe = Pipeline(
                steps=[
                    ('one_hot',OneHotEncoder()),
                    ('scaler',RobustScaler())
                ]
            )

            logging.info('Categorical columns encodering completed')

            state_pipe = Pipeline(
                steps=[
                    ('ordinal',OrdinalEncoder()),
                    ('scaler',RobustScaler())
                ]
            )

            logging.info('State column encodering completed')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipe,numerical_columns),
                    ('cat_pipeline',cat_pipe,categorical_columns),
                    ('state_pipeline',state_pipe,state_columns)],remainder='drop'
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('The train and test dataset loaded')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'churn'

            feature_cols = ['churn','area_code','account_length']

            input_feature_train_df = train_df.drop(columns=feature_cols,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=feature_cols,axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training and testing dataframes')

            final_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessing_obj),
                ('smote', SMOTE())
            ])

            # input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            # train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            # test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            input_feature_train_arr, target_feature_train_arr = final_pipeline.fit_resample(input_feature_train_df, target_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.values]

            logging.info('Saved preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj # save columnstransformater
            )

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path_full_pipeline,
                obj = final_pipeline # save combined full pipeline with SMOT
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)