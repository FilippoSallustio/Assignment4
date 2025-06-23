import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 1. Load data
file_path = 'dataftsemib_manual.csv'

df = pd.read_csv(file_path)

# 2. Clean data
# Parse dates
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Numeric columns with commas
num_cols = ['Price', 'Open', 'High', 'Low']
for col in num_cols:
    df[col] = df[col].str.replace(',', '')
    df[col] = df[col].astype(float)

# Volume column (M/B suffix)
def parse_volume(v):
    v = str(v).strip()
    if v.endswith('M'):
        return float(v[:-1].replace(',', '')) * 1e6
    if v.endswith('B'):
        return float(v[:-1].replace(',', '')) * 1e9
    return float(v.replace(',', ''))

df['Vol.'] = df['Vol.'].apply(parse_volume)

# Change percent column
change = df['Change %'].str.replace('%', '').str.replace(',', '.')
df['Change %'] = change.astype(float)

# Sort by date and reset index
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# 3. Use closing price
prices = df['Price']

# Train/test split
split = int(len(prices) * 0.8)
train, test = prices[:split], prices[split:]

# 4. Fit ARIMA model (simple order)
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# 5. Forecast on test set
forecast = model_fit.forecast(steps=len(test))

rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mape = mean_absolute_percentage_error(test, forecast)
r2 = r2_score(test, forecast)

print(
    f"Test RMSE: {rmse:.2f}\n"
    f"Test MAE: {mae:.2f}\n"
    f"Test MAPE: {mape:.2%}\n"
    f"Test R^2: {r2:.2f}"
)

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(df['Date'].iloc[split:], test.values, label='Actual')
plt.plot(df['Date'].iloc[split:], forecast.values, label='Predicted')
plt.xlabel('Date')
plt.ylabel('FTSE MIB Close')
plt.legend()
plt.tight_layout()
plt.savefig('arima_prediction_plot.png')

# 6. Forecast next day
model_full = ARIMA(prices, order=(5,1,0))
model_full_fit = model_full.fit()
next_price = model_full_fit.forecast().iloc[0]
print(f"Next day predicted close: {next_price:.2f}")
