from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
ROOT = Path(__file__).resolve().parent
st.title('Monthly Rainfall Prediction')
st.warning('Synthetic demonstration only. Not a real weather forecast.')
path = ROOT / 'models/model.joblib'
if not path.exists():
    st.error('Run python train.py first.')
    st.stop()
a = joblib.load(path)
month = st.selectbox('Target month', list(range(1,13)))
lag1 = st.number_input('Previous month rainfall (mm)', min_value=0.0, value=50.0)
lag12 = st.number_input('Same month last year rainfall (mm)', min_value=0.0, value=80.0)
if st.button('Predict'):
    st.metric('Estimated rainfall (mm)', f"{a['model'].predict(pd.DataFrame([{'month':month,'lag1':lag1,'lag12':lag12}]))[0]:.1f}")
st.write(a['metrics'])
st.line_chart(pd.read_csv(ROOT / 'reports/predictions.csv', parse_dates=['date']).set_index('date')[['actual','predicted']])
