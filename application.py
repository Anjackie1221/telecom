from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application = Flask(__name__)
app = application

## route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            state=request.form.get('state'),
            account_length=int(request.form.get('account_length')),
            area_code=int(request.form.get('area_code')),
            international_plan=request.form.get('international_plan'),
            voice_mail_plan=request.form.get('voice_mail_plan'),
            number_vmail_messages=int(request.form.get('number_vmail_messages')),
            total_day_minutes=float(request.form.get('total_day_minutes')),
            total_day_calls=int(request.form.get('total_day_calls')),
            total_day_charge=float(request.form.get('total_day_charge')),
            total_eve_minutes=float(request.form.get('total_eve_minutes')),
            total_eve_calls=int(request.form.get('total_eve_calls')),
            total_eve_charge=float(request.form.get('total_eve_charge')),
            total_night_minutes=float(request.form.get('total_night_minutes')),
            total_night_calls=int(request.form.get('total_night_calls')),
            total_night_charge=float(request.form.get('total_night_charge')),
            total_intl_minutes=float(request.form.get('total_intl_minutes')),
            total_intl_calls=int(request.form.get('total_intl_calls')),
            total_intl_charge=float(request.form.get('total_intl_charge')),
            customer_service_calls=int(request.form.get('customer_service_calls'))
        )

    pred_df = data.get_data_as_data_frame()
    print(pred_df)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return render_template('home.html',results=results[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0')

