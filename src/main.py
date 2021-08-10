import plotly.graph_objects as go
import math
from sklearn.metrics import mean_squared_error
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from preprocess_data import preprocess
from LSTM_model import train_LSTM
from GRU_model import train_GRU


def main():
    x_train, x_test, y_train_lstm, y_test_lstm, y_train_gru, y_test_gru, scaler, price, lookback = preprocess()
    lstm = train_LSTM(x_train, x_test, y_train_lstm, y_test_lstm,
                      y_train_gru, y_test_gru, scaler, price, lookback)
    gru = train_GRU(x_train, x_test, y_train_lstm, y_test_lstm,
                    y_train_gru, y_test_gru, scaler, price, lookback)

    lstm = pd.DataFrame(lstm, columns=['LSTM'])
    gru = pd.DataFrame(gru, columns=['GRU'])
    result = pd.concat([lstm, gru], axis=1, join='inner')
    result.index = ['Train RMSE', 'Test RMSE', 'Train Time']
    print()
    print(result)
    print()


main()
