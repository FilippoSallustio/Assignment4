"""ARIMA forecasting of the FTSE MIB closing prices.

This script loads the dataset, cleans numeric columns, plots the series,
checks for stationarity with the Dickey-Fuller test, visualizes ACF and
PACF, then fits an ARIMA model selected via a simple grid search. It evaluates the
forecast on an 80/20 train-test split and saves plots of the data and
predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools

# 1. Load data
file_path = "dataftsemib_manual.csv"

df = pd.read_csv(file_path)

# 2. Clean data
# Parse dates
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Numeric columns with commas
num_cols = ["Price", "Open", "High", "Low"]
for col in num_cols:
    df[col] = df[col].str.replace(",", "")
    df[col] = df[col].astype(float)

# Volume column (M/B suffix)

def parse_volume(v: str) -> float:
    v = str(v).strip()
    if v.endswith("M"):
        return float(v[:-1].replace(",", "")) * 1e6
    if v.endswith("B"):
        return float(v[:-1].replace(",", "")) * 1e9
    return float(v.replace(",", ""))


df["Vol."] = df["Vol."].apply(parse_volume)

# Change percent column
change = df["Change %"].str.replace("%", "").str.replace(",", ".")
df["Change %"] = change.astype(float)

# Sort by date and reset index

df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

# 3. Plot the cleaned closing prices
plt.figure(figsize=(10, 4))
plt.plot(df["Date"], df["Price"])
plt.title("FTSE MIB Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("arima_cleaned_prices.png")
plt.show()
plt.close()

# 4. Stationarity check
result = adfuller(df["Price"])
print("Dickey-Fuller p-value:", result[1])

# 5. ACF and PACF of first-differenced series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df["Price"].diff().dropna(), ax=axes[0], lags=40)
axes[0].set_title("ACF")
plot_pacf(df["Price"].diff().dropna(), ax=axes[1], lags=40, method="ywm")
axes[1].set_title("PACF")
plt.tight_layout()
plt.savefig("arima_acf_pacf.png")
plt.show()
plt.close()

# 6. Train/test split
prices = df["Price"]
split = int(len(prices) * 0.8)
train, test = prices[:split], prices[split:]

# 7. Search for a good (p,d,q) order using AIC
p = range(0, 4)
d = [0, 1]
q = range(0, 4)
best_aic = float("inf")
best_order = (1, 1, 1)
for order in itertools.product(p, d, q):
    try:
        model = ARIMA(train, order=order)
        result = model.fit()
        if result.aic < best_aic:
            best_aic = result.aic
            best_order = order
    except Exception:
        continue

print("Selected order:", best_order)
model_fit = ARIMA(train, order=best_order).fit()

# 8. Forecast on test set
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
plt.figure(figsize=(10, 5))
plt.plot(df["Date"].iloc[split:], test.values, label="Actual")
plt.plot(df["Date"].iloc[split:], forecast.values, label="Predicted")
plt.xlabel("Date")
plt.ylabel("FTSE MIB Close")
plt.legend()
plt.tight_layout()
plt.savefig("arima_prediction_plot.png")
plt.show()
plt.close()

# 9. Forecast next day using entire dataset
final_model = ARIMA(prices, order=best_order)
final_fit = final_model.fit()
next_price = final_fit.forecast().iloc[0]
print(f"Next day predicted close: {next_price:.2f}")
