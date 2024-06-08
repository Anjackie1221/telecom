import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

st.write("## This is the Telecom Churn Prediction")

st.sidebar.header("User Input")

st.sidebar.markdown ("""
[Example CSV input file](https://github.com/Anjackie1221/telecom/blob/main/notebook/data/churn-bigml-20.csv
""")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv" ])
if uploaded_file is not None:
    pred_df = pd. read_csv(uploaded_file)
else:
    data = CustomData(
        state = st.sidebar.selectbox( 'State', ('NY', 'KS', 'CA','MI','NJ','CO','AZ' )),
        account_length = st.sidebar.slider('account_length',10,300,22),
        area_code = st.sidebar.slider('area_code',400,500,408),
        international_plan = st.sidebar.selectbox( 'international_plan', ('Yes', 'No')),
        voice_mail_plan = st.sidebar.selectbox( 'voice_mail_plan', ('Yes', 'No')),
        number_vmail_messages = st.sidebar.slider('number_vmail_messages',0,200,184),
        total_day_minutes = float(st.sidebar.text_input('total_day_minutes', 43.9)),
        total_day_calls = st.sidebar.slider('total_day_calls', 1,300,97),
        total_day_charge = float(st.sidebar.text_input('total_day_charges', 31.37)),
        total_eve_minutes = float(st.sidebar.text_input('total_eve_minutes', 351.9)),
        total_eve_calls = st.sidebar.slider('total_eve_calls', 1,300,80),
        total_eve_charge = float(st.sidebar.text_input('total_eve_charges',329.89)),
        total_night_minutes = float(st.sidebar.text_input('total_night_minutes',251.8)),
        total_night_calls = st.sidebar.slider('total_night_calls', 1,300,90),
        total_night_charge = float(st.sidebar.text_input('total_night_charges',39.71)),
        total_intl_minutes = float(st.sidebar.text_input('total_intl_minutes', 8.7)),
        total_intl_calls = st.sidebar.slider('total_intl_calls', 10,500,4),
        total_intl_charge = float(st.sidebar.text_input('total_intl_charge',2.35)),
        customer_service_calls = st.sidebar.slider('customer_service_calls', 1,100,1)
        )
    pred_df = data.get_data_as_data_frame()

st.subheader ('User Input features')

if uploaded_file is not None:
    st.write(pred_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters')
    st.write(pred_df)

predict_pipeline = PredictPipeline()
results = predict_pipeline.predict(pred_df)

st.subheader('Prediction')

st.write(results)
