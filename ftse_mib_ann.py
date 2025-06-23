"""LSTM-based forecast of the FTSE MIB closing price.

This script walks through loading the dataset, cleaning it, preparing
sequences of closing prices and fitting a small neural network to
forecast future values.
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

tf.random.set_seed(0)

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
    if v.endswith('M'):
        return float(v[:-1].replace(',', '')) * 1e6
    if v.endswith('B'):
        return float(v[:-1].replace(',', '')) * 1e9
    return float(v.replace(',', ''))

df['Vol.'] = df['Vol.'].apply(parse_volume)

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

# 3. Prepare features for ANN
# We'll predict closing price using the previous 60 closing prices
close_values = df[['Price']].values
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(close_values)

window_size = 60
X, y = [], []
for i in range(window_size, len(scaled_close)):
    X.append(scaled_close[i-window_size:i, 0])
    y.append(scaled_close[i, 0])
X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Build the LSTM network
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 5. Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=2
)

# 6. Evaluate on test set
pred = model.predict(X_test)
# Inverse scale
pred_prices = scaler.inverse_transform(pred)
true_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

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

# 7. Forecast the next day
last_window = scaled_close[-window_size:]
next_pred = model.predict(last_window.reshape(1, window_size, 1))
next_price = scaler.inverse_transform(next_pred)[0,0]
print(f"Next day predicted close: {next_price:.2f}")

# Save the model for later use
model.save("ftse_mib_ann_model.h5")

