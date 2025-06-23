"""LSTM-based forecast of the FTSE MIB closing price.

This script walks through loading the dataset, cleaning it and
preparing sequences of recent prices along with a few auxiliary
features.  A two-layer LSTM network is trained with early stopping to
forecast the next closing value.  The script saves plots of the
cleaned data and of the prediction quality.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

tf.random.set_seed(0)

# parse command line arguments so tests can shorten training
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
args = parser.parse_args()

# 1. Load data
file_path = 'dataftsemib_manual.csv'

df = pd.read_csv(file_path)

# 2. Clean data
# Parse dates
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Remove comma thousand separators from numeric columns
num_cols = ['Price', 'Open', 'High', 'Low']
for col in num_cols:
    df[col] = df[col].str.replace(',', '')
    df[col] = df[col].astype(float)

# Parse volume column (handle millions/billions)
def parse_volume(v):
    v = str(v).strip()
    if v == "-":
        return np.nan
    if v.endswith("M"):
        return float(v[:-1].replace(",", "")) * 1e6
    if v.endswith("B"):
        return float(v[:-1].replace(",", "")) * 1e9
    return float(v.replace(",", ""))

df['Vol.'] = df['Vol.'].apply(parse_volume)
df['Vol.'].fillna(method="ffill", inplace=True)

# Convert change percentage
# Sometimes comma is used as decimal separator - handle it
change = df['Change %'].str.replace('%', '').str.replace(',', '.')
df['Change %'] = change.astype(float)

# Sort by date
df.sort_values('Date', inplace=True)

df.reset_index(drop=True, inplace=True)

# Plot the cleaned closing price data to verify preprocessing
plt.figure(figsize=(10, 4))
plt.plot(df['Date'], df['Price'])
plt.title('FTSE MIB Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.tight_layout()
plt.savefig('cleaned_data_plot.png')
plt.show()

# 3. Prepare features for ANN
# We'll use the last 60 observations of several columns to predict the
# next closing price.  Scaling is applied to all features at once so
# their magnitudes are comparable.

feature_cols = ["Price", "Open", "High", "Low", "Vol."]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[feature_cols])

window_size = 60
X, y = [], []
for i in range(window_size, len(scaled)):
    X.append(scaled[i - window_size : i])
    # the close price is the first column in feature_cols
    y.append(scaled[i, 0])
X = np.array(X)
y = np.array(y)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Build the LSTM network
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(window_size, len(feature_cols))),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# 5. Train the model
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
history = model.fit(
    X_train,
    y_train,
    epochs=args.epochs,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=2,
)

# 6. Evaluate on test set
pred = model.predict(X_test)
# Inverse scale only on the price column
pred_full = np.zeros((len(pred), len(feature_cols)))
true_full = np.zeros((len(y_test), len(feature_cols)))
pred_full[:, 0] = pred[:, 0]
true_full[:, 0] = y_test
pred_prices = scaler.inverse_transform(pred_full)[:, 0]
true_prices = scaler.inverse_transform(true_full)[:, 0]

rmse = np.sqrt(mean_squared_error(true_prices, pred_prices))
mae = mean_absolute_error(true_prices, pred_prices)
mape = mean_absolute_percentage_error(true_prices, pred_prices)
r2 = r2_score(true_prices, pred_prices)
print(
    f"Test RMSE: {rmse:.2f}\n"
    f"Test MAE: {mae:.2f}\n"
    f"Test MAPE: {mape:.2%}\n"
    f"Test R^2: {r2:.2f}"
)

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(df['Date'].iloc[split+window_size:], true_prices.flatten(), label='Actual')
plt.plot(df['Date'].iloc[split+window_size:], pred_prices.flatten(), label='Predicted')
plt.xlabel('Date')
plt.ylabel('FTSE MIB Close')
plt.legend()
plt.tight_layout()
plt.savefig('prediction_plot.png')
plt.show()

# 7. Forecast the next day using the most recent window
last_window = scaled[-window_size:]
next_pred = model.predict(last_window.reshape(1, window_size, len(feature_cols)))
next_full = np.zeros((1, len(feature_cols)))
next_full[:, 0] = next_pred[:, 0]
next_price = scaler.inverse_transform(next_full)[0, 0]
print(f"Next day predicted close: {next_price:.2f}")

# Save the model for later use
model.save("ftse_mib_ann_model.h5")

