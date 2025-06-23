"""EGARCH forecasting of FTSE MIB returns.

This script loads the FTSE MIB dataset, cleans it, computes log returns
and visualizes them. It then tests for stationarity with the
Dickey-Fuller test and checks for ARCH effects. An EGARCH model is
selected via a small grid search on the training portion (80% of the
data). A rolling one-step forecast evaluates performance on the test
set and a next-day return forecast is produced.
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
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
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
df["Vol."].fillna(method="ffill", inplace=True)

# Change percent column
change = df["Change %"].str.replace("%", "").str.replace(",", ".")
df["Change %"] = change.astype(float)

# Sort by date and reset index

df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

# 3. Calculate log returns (percent)
df["Return"] = np.log(df["Price"]).diff() * 100
returns = df["Return"].dropna()
return_dates = df["Date"].iloc[1:]

# Plot returns
plt.figure(figsize=(10, 4))
plt.plot(return_dates, returns)
plt.title("FTSE MIB Daily Log Returns (%)")
plt.xlabel("Date")
plt.ylabel("Return %")
plt.tight_layout()
plt.savefig("returns_plot.png")
plt.show()
plt.close()

# 4. Stationarity test
adf_stat, adf_p, *_ = adfuller(returns)
print(f"Dickey-Fuller p-value: {adf_p:.4f}")

# 5. ARCH effect test
lm_stat, lm_p, _, _ = het_arch(returns)
print(f"ARCH LM-test p-value: {lm_p:.4f}")

# 6. Train/test split
split = int(len(returns) * 0.8)
train, test = returns[:split], returns[split:]

# 7. Grid search for EGARCH(p,o,q) order using AIC
p_range = [1, 2]
o_range = [1]
q_range = [1, 2]
best_order = (1, 1, 1)
best_aic = float("inf")
for order in itertools.product(p_range, o_range, q_range):
    try:
        am = arch_model(
            train,
            mean="AR",
            lags=1,
            vol="EGARCH",
            p=order[0],
            o=order[1],
            q=order[2],
            dist="normal",
        )
        res = am.fit(disp="off")
        if res.aic < best_aic:
            best_aic = res.aic
            best_order = order
    except Exception:
        continue
print("Selected order:", best_order)

# 8. Rolling one-step forecast on test set
history = train.copy()
forecast_vals = []
for obs in test:
    am = arch_model(
        history,
        mean="AR",
        lags=1,
        vol="EGARCH",
        p=best_order[0],
        o=best_order[1],
        q=best_order[2],
        dist="normal",
    )
    res = am.fit(disp="off")
    fc = res.forecast(horizon=1)
    forecast_vals.append(fc.mean.iloc[-1, 0])
    history = pd.concat([history, pd.Series([obs])], ignore_index=True)

forecast = pd.Series(forecast_vals, index=test.index)

rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mape = mean_absolute_percentage_error(test, forecast)
r2 = r2_score(test, forecast)
print(
    f"Test RMSE: {rmse:.4f}\n"
    f"Test MAE: {mae:.4f}\n"
    f"Test MAPE: {mape:.2%}\n"
    f"Test R^2: {r2:.4f}"
)

# Plot actual vs predicted returns
plt.figure(figsize=(10, 5))
plt.plot(return_dates.iloc[split:], test.values, label="Actual")
plt.plot(return_dates.iloc[split:], forecast.values, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Return %")
plt.legend()
plt.tight_layout()
plt.savefig("egarch_prediction_plot.png")
plt.show()
plt.close()

# 9. Forecast next day return using entire dataset
final_model = arch_model(
    returns,
    mean="AR",
    lags=1,
    vol="EGARCH",
    p=best_order[0],
    o=best_order[1],
    q=best_order[2],
    dist="normal",
)
final_res = final_model.fit(disp="off")
next_fc = final_res.forecast(horizon=1)
next_ret = next_fc.mean.iloc[-1, 0]
print(f"Next day predicted return (%): {next_ret:.4f}")
