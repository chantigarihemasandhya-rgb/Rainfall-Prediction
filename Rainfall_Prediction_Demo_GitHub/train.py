from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
ROOT = Path(__file__).resolve().parent
FEATURES = ['month', 'lag1', 'lag12']
def train():
    df = pd.read_csv(ROOT / 'data/synthetic_monthly_rainfall.csv', parse_dates=['date']).sort_values('date')
    df['month'] = df.date.dt.month
    df['lag1'] = df.rainfall_mm.shift(1)
    df['lag12'] = df.rainfall_mm.shift(12)
    df = df.dropna()
    train, test = df.iloc[:-24], df.iloc[-24:]
    model = RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=3)
    model.fit(train[FEATURES], train.rainfall_mm)
    pred = model.predict(test[FEATURES])
    report = pd.DataFrame({'date':test.date, 'actual':test.rainfall_mm, 'predicted':pred})
    for folder in ['models','reports']:
        (ROOT / folder).mkdir(exist_ok=True)
    report.to_csv(ROOT / 'reports/predictions.csv', index=False)
    metrics = {'MAE':mean_absolute_error(test.rainfall_mm,pred), 'RMSE':mean_squared_error(test.rainfall_mm,pred)**0.5, 'seasonal_baseline_MAE':mean_absolute_error(test.rainfall_mm,test.lag12)}
    joblib.dump({'model':model,'metrics':metrics}, ROOT / 'models/model.joblib')
    print(metrics)
if __name__ == '__main__':
    train()
