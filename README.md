# FTSE MIB Forecasting

This repository demonstrates how to forecast the FTSE MIB index using
an LSTM-based neural network.
The data is provided in `dataftsemib_manual.csv`.

## Prerequisites

Install the required Python packages:

```bash
pip install pandas matplotlib scikit-learn tensorflow==2.12.0 statsmodels pmdarima
```

## Running the ANN example

Execute the ANN script to clean the data, train the model and produce a
one-step ahead forecast:

```bash
python3 ftse_mib_ann.py --epochs 50  # adjust epochs as needed
```

The script uses not only the closing price but also open, high, low and
volume information.  Training stops automatically when the validation
loss does not improve for several epochs.  It displays and also saves
two plots:

- `cleaned_data_plot.png` – the closing prices after preprocessing
- `prediction_plot.png` – comparison of actual vs predicted prices

A trained model file `ftse_mib_ann_model.h5` is also saved.

## ARIMA example

For a simple statistical baseline you can also run:

```bash
python3 ftse_mib_arima.py
```

This script displays and saves several diagnostic plots including the cleaned closing
prices (`arima_cleaned_prices.png`) as well as ACF/PACF graphs
(`arima_acf_pacf.png`). It then fits an ARIMA model selected via a small grid
search and produces a rolling one-step forecast to compare against the test
data. The series is split 80/20 between training and test portions. The results
are saved to `arima_prediction_plot.png`. If the residuals of the initial model
exhibit autocorrelation (Ljung-Box p < 0.05), the script automatically
re-estimates an ARIMA(3,1,3) model for comparison.


## EGARCH example

To model the volatility of daily returns you can run the EGARCH
script:

```bash
python3 ftse_mib_egarch.py
```

The script calculates log returns and uses log volume as an exogenous
regressor. After checking stationarity and ARCH effects it searches a
small grid of EGARCH orders and forecasts conditional variance rather
than returns.  The 80/20 train/test split is evaluated using a
range-based volatility proxy.  Forecasts are displayed and saved as
`egarch_variance_plot.png`.
