import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            # full_pipeline_path = 'artifacts/full_pipeline.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            print("Before Loading")
            model = load_object(file_path = model_path)
            pipe = load_object(file_path = preprocessor_path)
            print("After Loading")
            data_scaled = pipe.transform(features)
            pres = model.predict(data_scaled)
            return pres
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
        state: str,
        account_length: int,
        area_code: int,
        international_plan: str,
        voice_mail_plan: str,
        number_vmail_messages: int,
        total_day_minutes: float,
        total_day_calls: int,
        total_day_charge: float,
        total_eve_minutes: float,
        total_eve_calls: int,
        total_eve_charge: float,
        total_night_minutes: float,
        total_night_calls: int,
        total_night_charge: float,
        total_intl_minutes: float,
        total_intl_calls: int,
        total_intl_charge: float,
        customer_service_calls: int):

        self.state = state
        self.account_length = account_length
        self.area_code = area_code
        self.international_plan = international_plan
        self.voice_mail_plan = voice_mail_plan
        self.number_vmail_messages = number_vmail_messages
        self.total_day_minutes = total_day_minutes
        self.total_day_calls = total_day_calls
        self.total_day_charge = total_day_charge
        self.total_eve_minutes = total_eve_minutes
        self.total_eve_calls = total_eve_calls
        self.total_eve_charge = total_eve_charge
        self.total_night_minutes = total_night_minutes
        self.total_night_calls = total_night_calls
        self.total_night_charge = total_night_charge
        self.total_intl_minutes = total_intl_minutes
        self.total_intl_calls = total_intl_calls
        self.total_intl_charge = total_intl_charge
        self.customer_service_calls = customer_service_calls

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "state": [self.state],
                "account_length": [self.account_length],
                "area_code": [self.area_code],
                "international_plan": [self.international_plan],
                "voice_mail_plan": [self.voice_mail_plan],
                "number_vmail_messages": [self.number_vmail_messages],
                "total_day_minutes": [self.total_day_minutes],
                "total_day_calls": [self.total_day_calls],
                "total_day_charge": [self.total_day_charge],
                "total_eve_minutes": [self.total_eve_minutes],
                "total_eve_calls": [self.total_eve_calls],
                "total_eve_charge": [self.total_eve_charge],
                "total_night_minutes": [self.total_night_minutes],
                "total_night_calls": [self.total_night_calls],
                "total_night_charge": [self.total_night_charge],
                "total_intl_minutes": [self.total_intl_minutes],
                "total_intl_calls": [self.total_intl_calls],
                "total_intl_charge": [self.total_intl_charge],
                "customer_service_calls": [self.customer_service_calls]
            }

            pd.DataFrame(custom_data_input_dict).to_csv('input_values.csv')
            return pd.DataFrame(custom_data_input_dict)
            
        
        except Exception as e:
            raise CustomException(e,sys)