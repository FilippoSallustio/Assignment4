# FTSE MIB Forecasting

This repository demonstrates how to forecast the FTSE MIB index using an LSTM-based neural network.
The data is provided in `dataftsemib_manual.csv`.

## Prerequisites

Install the required Python packages:

```bash
pip install pandas matplotlib scikit-learn tensorflow==2.12.0 statsmodels pmdarima
```

## Running the ANN example

Execute the script to clean the data, train the model and produce a one-step ahead forecast:

```bash
python3 ftse_mib_ann.py
```

The script saves two plots:

- `cleaned_data_plot.png` – the closing prices after preprocessing
- `prediction_plot.png` – comparison of actual vs predicted prices

A trained model file `ftse_mib_ann_model.h5` is also saved.

## ARIMA example

For a simple statistical baseline you can also run:

```bash
python3 ftse_mib_arima.py
```

This script produces several diagnostic plots including the cleaned closing
prices (`arima_cleaned_prices.png`) as well as ACF/PACF graphs
(`arima_acf_pacf.png`). It then fits an ARIMA model chosen via `auto_arima` and
saves the forecast comparison in `arima_prediction_plot.png`.

