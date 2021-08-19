import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def to_days(then):
    now = datetime.date.today()
    date_time_obj = datetime.datetime.strptime(then, '%Y-%m-%d').date()
    diff = (now - date_time_obj)
    diff = str(diff).split(' ')
    return int(diff[0])


def load_data():
    filepath = '../dataset/TSLA.csv'
    df = pd.read_csv(filepath)
    return df


def load_plot():

    # load dataset
    df = load_data()

    # get days since stock price

    df['Date'] = df['Date'].astype(str)
    df['Days'] = df['Date']
    row = 0
    for i in df['Date']:
        df.loc[row, 'Days'] = to_days(df['Date'][row])
        row += 1
    print(df.head())
    print()
    df = df.drop('Date', axis=1)
    return df



    # historical view of the closing price

    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 9))
    plt.plot(df[['Close']])
    plt.xticks(range(0, df.shape[0], 250), df['Date'].loc[::250], rotation=45)
    plt.title("Closing Price of TESLA", fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price (USD)', fontsize=18)
    plt.tight_layout()
    plt.show()

    # volume of stock being traded each day

    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 9))
    plt.plot(df[['Volume']])
    plt.xticks(range(0, df.shape[0], 250), df['Date'].loc[::250], rotation=45)
    plt.title("Sales Volume for TESLA", fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Volume', fontsize=18)
    plt.tight_layout()
    plt.show()

    # daily return

    df['Daily Return'] = df['Adj Close'].pct_change()
    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 9))
    plt.plot(df[['Daily Return']])
    plt.xticks(range(0, df.shape[0], 250), df['Date'].loc[::250], rotation=45)
    plt.title("Daily Return for TESLA", fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Daily Return', fontsize=18)
    plt.tight_layout()
    plt.show()

    # moving average

    ma_day = [10, 20, 50]
    for ma in ma_day:
        column_name = f"MA for {ma} days"
        df[column_name] = df['Adj Close'].rolling(ma).mean()
    df[['Adj Close', 'MA for 10 days', 'MA for 20 days',
        'MA for 50 days']].plot(figsize=(15, 9))
    plt.xticks(range(0, df.shape[0], 250), df['Date'].loc[::250], rotation=45)
    plt.title("TESLA", fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=18)
    plt.tight_layout()
    plt.show()

    
