import pandas as pd
from preprocess_data import preprocess
from LSTM_model import train_LSTM
from GRU_model import train_GRU
from MLP_model import train_MLP
from CNN_model import train_CNN


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, scaler, price, lookback = preprocess()

    cnn = train_CNN(x_train, x_test, y_train,
                    y_test, scaler, price, lookback)

    lstm = train_LSTM(x_train, x_test, y_train,
                      y_test, scaler, price, lookback)
    gru = train_GRU(x_train, x_test, y_train, y_test, scaler, price, lookback)

    mlp = train_MLP(x_train, x_test, y_train, y_test, scaler, price, lookback)

    lstm = pd.DataFrame(lstm, columns=['LSTM'])
    gru = pd.DataFrame(gru, columns=['GRU'])
    mlp = pd.DataFrame(mlp, columns=['MLP'])
    cnn = pd.DataFrame(cnn, columns=['CNN'])

    result = pd.concat([lstm, gru, mlp], axis=1, join='inner')
    result.index = ['Train RMSE', 'Test RMSE', 'Train Time']
    print()
    print(result)
    print()
