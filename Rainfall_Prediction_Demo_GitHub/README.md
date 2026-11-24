# Monthly Rainfall Prediction

A standalone learning project using Python, Random Forest regression and Streamlit. This is a newly prepared example, not a copy of any existing GitHub repository.

## Data
The included 300 monthly observations (2000–2024) are synthetic, generated with seasonal patterns and random variation. They do not represent measured rainfall in India or any real location. Do not use for agricultural or weather decisions.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```
Model and reports are generated locally. The model uses month, previous-month rainfall and rainfall from 12 months earlier. The final 24 months form a chronological holdout; metrics include MAE, RMSE and seasonal baseline MAE. Evaluation assumes observed historical lag values are available for each prediction, not a recursive 24-month forecast.
