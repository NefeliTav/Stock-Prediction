import plotly.graph_objects as go
import math
from sklearn.metrics import mean_squared_error
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

input_dim = 1
hidden_dim = 32
num_layers = 4
output_dim = 1
num_epochs = 200


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.act(hidden)
        output = self.fc2(relu[:, -1, :])
        return output


def train_MLP(x_train, x_test, y_train, y_test, scaler, price, lookback):

    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
                output_dim=output_dim, num_layers=num_layers)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss(reduction='mean')

    hist = np.zeros(num_epochs)
    start_time = time.time()
    mlp = []

    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print()
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))

    original = pd.DataFrame(scaler.inverse_transform(
        y_train.detach().numpy()))

    # make predictions
    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(
        y_train[:, 0], y_train_pred[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    mlp.append(trainScore)
    mlp.append(testScore)
    mlp.append(training_time)

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

    original = scaler.inverse_transform(price['Close'].values.reshape(-1, 1))

    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)

    fig = go.Figure()
    fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                                        mode='lines',
                                        name='Train prediction')))
    fig.add_trace(go.Scatter(x=result.index, y=result[1],
                             mode='lines',
                             name='Test prediction'))
    fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                                        mode='lines',
                                        name='Actual Value')))
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='white',
            linewidth=2
        ),
        yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
        ),
        showlegend=True,
        template='plotly_dark'

    )

    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                            xanchor='left', yanchor='bottom',
                            text='Results (MLP)',
                            font=dict(family='Rockwell',
                                      size=26,
                                      color='white'),
                            showarrow=False))
    fig.update_layout(annotations=annotations)

    fig.show()

    return mlp
