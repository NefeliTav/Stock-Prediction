import pandas as pd
from preprocess_data import preprocess
from LSTM_model import train_LSTM
from GRU_model import train_GRU
from MLP_model import train_MLP


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, scaler, price, lookback = preprocess()
    '''
    lstm = train_LSTM(x_train, x_test, y_train, y_test,
                      y_train_gru, y_test_gru, scaler, price, lookback)
    gru = train_GRU(x_train, x_test, y_train, y_test,
                    y_train_gru, y_test_gru, scaler, price, lookback)

    lstm = pd.DataFrame(lstm, columns=['LSTM'])
    gru = pd.DataFrame(gru, columns=['GRU'])
    result = pd.concat([lstm, gru], axis=1, join='inner')
    result.index = ['Train RMSE', 'Test RMSE', 'Train Time']
    print()
    print(result)
    print()
    '''
    mlp = train_MLP(x_train, x_test, y_train, y_test, scaler, price, lookback)
