"""EGARCH volatility forecasting for the FTSE MIB.

This script cleans the data and computes daily log returns.  It tests for
stationarity and ARCH effects before selecting an EGARCH specification via
a small grid search.  The model uses log volume as an exogenous regressor
in the mean equation and forecasts conditional variance rather than the
return itself.  Forecasted variance is evaluated against a range-based
volatility proxy on an 80/20 split of the data.
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
# Additional features for volatility modelling
df["LogVol"] = np.log(df["Vol."])
df["Range"] = (np.log(df["High"]) - np.log(df["Low"])) * 100

# Drop initial NaN from return difference
df.dropna(inplace=True)

returns = df["Return"]
exog = df[["LogVol"]]
return_dates = df["Date"]

# Realized variance proxy combining squared returns and range
df["RealizedVar"] = 0.5 * df["Return"].pow(2) + 0.5 * df["Range"].pow(2)

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
exog_train, exog_test = exog[:split], exog[split:]
rv_train, rv_test = df["RealizedVar"][:split], df["RealizedVar"][split:]

# 7. Grid search for EGARCH(p,o,q) order using AIC
p_range = [1, 2, 3]
o_range = [1, 2]
q_range = [1, 2, 3]
best_order = (1, 1, 1)
best_aic = float("inf")
for order in itertools.product(p_range, o_range, q_range):
    try:
        am = arch_model(
            train,
            x=exog_train,
            mean="ARX",
            lags=1,
            vol="EGARCH",
            p=order[0],
            o=order[1],
            q=order[2],
            dist="t",
        )
        res = am.fit(disp="off")
        if res.aic < best_aic:
            best_aic = res.aic
            best_order = order
    except Exception:
        continue
print("Selected order:", best_order)

# 8. Rolling one-step forecast on test set
history_y = train.copy()
history_x = exog_train.copy()
forecast_var = []
for i in range(len(test)):
    am = arch_model(
        history_y,
        x=history_x,
        mean="ARX",
        lags=1,
        vol="EGARCH",
        p=best_order[0],
        o=best_order[1],
        q=best_order[2],
        dist="t",
    )
    res = am.fit(disp="off")
    fc = res.forecast(horizon=1, x=exog_test.iloc[i:i+1])
    forecast_var.append(fc.variance.iloc[-1, 0])
    history_y = pd.concat([history_y, pd.Series([test.iloc[i]])], ignore_index=True)
    history_x = pd.concat([history_x, exog_test.iloc[i:i+1]], ignore_index=True)

forecast = pd.Series(forecast_var, index=test.index)

rmse = np.sqrt(mean_squared_error(rv_test, forecast))
mae = mean_absolute_error(rv_test, forecast)
mape = mean_absolute_percentage_error(rv_test, forecast)
r2 = r2_score(rv_test, forecast)
qlike = np.mean(np.log(np.maximum(forecast, 1e-8)) + rv_test.values / np.maximum(forecast, 1e-8))
print(
    f"Test RMSE: {rmse:.4f}\n"
    f"Test MAE: {mae:.4f}\n"
    f"Test MAPE: {mape:.2%}\n"
    f"Test R^2: {r2:.4f}\n"
    f"Test QLIKE: {qlike:.4f}"
)

# Plot actual vs predicted conditional variance
plt.figure(figsize=(10, 5))
plt.plot(return_dates.iloc[split:], rv_test.values, label="Actual")
plt.plot(return_dates.iloc[split:], forecast.values, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Variance (%^2)")
plt.legend()
plt.tight_layout()
plt.savefig("egarch_variance_plot.png")
plt.show()
plt.close()

# 9. Forecast next day variance using entire dataset
final_model = arch_model(
    returns,
    x=exog,
    mean="ARX",
    lags=1,
    vol="EGARCH",
    p=best_order[0],
    o=best_order[1],
    q=best_order[2],
    dist="t",
)
final_res = final_model.fit(disp="off")
next_fc = final_res.forecast(horizon=1, x=exog.iloc[[-1]])
next_var = next_fc.variance.iloc[-1, 0]
print(
    f"Next day predicted variance (%^2): {next_var:.4f}\n"
    f"Model AIC: {final_res.aic:.2f}\n"
    f"Model BIC: {final_res.bic:.2f}\n"
    f"Model Log-Likelihood: {final_res.loglikelihood:.2f}"
)
