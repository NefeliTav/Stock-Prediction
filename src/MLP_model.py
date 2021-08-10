import plotly.graph_objects as go
import math
from sklearn.metrics import mean_squared_error
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Callable

num_epochs = 100


class MLP2D(nn.Module):
    def __init__(self,
                 num_layers: int,
                 hidden_dim: int,
                 activation: Callable[[torch.Tensor], torch.Tensor]
                 ) -> None:
        super().__init__()

        self.first_layer = nn.Linear(in_features=2,
                                     out_features=hidden_dim)

        # A list of modules: automatically exposes nested parameters to optimize.
        self.layers = nn.ModuleList()
        # Parameters contained in a normal python list are not returned by model.parameters()
        for i in range(num_layers):
            self.layers.append(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            )
        self.activation = activation

        self.last_layer = nn.Linear(in_features=hidden_dim,
                                    out_features=1)

    def forward(self, meshgrid: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to each (x, y) independently 

        :param meshgrid: tensor of dimensions [..., 2], where ... means any number of dims
        """
        out = meshgrid

        # First linear layer, transforms the hidden dimensions from 2 (the coordinates) to `hidden_dim`
        out = self.first_layer(out)
        for layer in self.layers:    # Apply `k` (linear, activation) layer
            out = layer(out)
            out = self.activation(out)
        # Last linear layer to bring the `hiddem_dim` features back to the 2 coordinates x, y
        out = self.last_layer(out)

        return out.squeeze(-1)


def train_MLP(x_train, x_test, y_train, y_test, scaler, price, lookback):
    model = MLP2D(num_layers=3,
                  hidden_dim=10,
                  activation=torch.nn.functional.relu)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []

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

    predict = pd.DataFrame(scaler.inverse_transform(
        y_train_pred.detach().numpy()))
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
    lstm.append(trainScore)
    lstm.append(testScore)
    lstm.append(training_time)

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
                            text='Results (LSTM)',
                            font=dict(family='Rockwell',
                                      size=26,
                                      color='white'),
                            showarrow=False))
    fig.update_layout(annotations=annotations)

    fig.show()

    return lstm
