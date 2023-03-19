import talib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import yfinance as yf
import numpy as np
from itertools import product


def get_accuracy(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    accuracy1 = accuracy_score(y_pred, y_train)
    print('in sample Accuracy:', accuracy1)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy2 = accuracy_score(y_test, y_pred)
    print('out of sample Accuracy:', accuracy2)
    return [accuracy1, accuracy2]

# Load data
#df = pd.read_csv('financial_data.csv')
#df = yf.download('SPY', start='2020-01-01')
#df = yf.download("AAPL", start="2020-01-01", end="2022-04-30")

#df = pd.read_csv("../temp_data.csv")
#df = yf.download("AAPL", start='2015-01-01', end='2023-03-18')
df = yf.download("SPY", start='2015-01-01', end='2023-03-18')

df['PivotPoint'] = (df['High'] + df['Low'] + df['Close'])/3
df['o1'] = df['Open'].shift(-1)
df['c1'] = df['Close'].shift(-1)
df['h1'] = df['High'].shift(-1)
df['l1'] = df['Low'].shift(-1)

# Split data

df['Target1'] = np.where( df['c1']- df['Close'] > 0, 1, -1 )
df['Target2'] = np.where( (df['o1'] - df['c1'])> 0, 1, -1 )
df['Target3'] = np.where( (df['h1'] - df['Close'])> 0, 1, -1 )
df['Target4'] = np.where( (df['l1'] - df['Close'])> 0, 1, -1 )
df['Target5'] = np.where( (df['l1'] - df['Low'])> 0, 1, -1 )
df['Target6'] = np.where( (df['h1'] - df['High'])> 0, 1, -1 )

df['Target7'] = np.where( (df['o1'] - df['Close'])> 0, 1, -1 )
df['Target8'] = np.where( (df['o1'] - df['Open'])> 0, 1, -1 )
df['Target9'] = np.where( (df['o1'] - df['High'])> 0, 1, -1 )
df['Target10'] = np.where( (df['o1'] - df['Low'])> 0, 1, -1 )

df['Target11'] = np.where( (df['h1'] - df['Close'])> 0, 1, -1 )
df['Target12'] = np.where( (df['h1'] - df['Open'])> 0, 1, -1 )
df['Target13'] = np.where( (df['h1'] - df['High'])> 0, 1, -1 )
df['Target14'] = np.where( (df['h1'] - df['Low'])> 0, 1, -1 )

df['Target15'] = np.where( (df['l1'] - df['Close'])> 0, 1, -1 )
df['Target16'] = np.where( (df['l1'] - df['Open'])> 0, 1, -1 )
df['Target17'] = np.where( (df['l1'] - df['High'])> 0, 1, -1 )
df['Target18'] = np.where( (df['l1'] - df['Low'])> 0, 1, -1 )

df['Target19'] = np.where( (df['c1'] - df['Close'])> 0, 1, -1 )
df['Target20'] = np.where( (df['c1'] - df['Open'])> 0, 1, -1 )
df['Target21'] = np.where( (df['c1'] - df['High'])> 0, 1, -1 )
df['Target22'] = np.where( (df['c1'] - df['Low'])> 0, 1, -1 )

df['Target23'] = np.where( (df['o1'] - df['PivotPoint'])> 0, 1, -1 )
df['Target24'] = np.where( (df['o1'] - df['PivotPoint'])> 0, 1, -1 )
df['Target25'] = np.where( (df['o1'] - df['PivotPoint'])> 0, 1, -1 )
df['Target26'] = np.where( (df['o1'] - df['PivotPoint'])> 0, 1, -1 )

df['Target27'] = np.where( (df['c1'] - df['PivotPoint'])> 0, 1, -1 )
df['Target28'] = np.where( (df['c1'] - df['PivotPoint'])> 0, 1, -1 )
df['Target29'] = np.where( (df['c1'] - df['PivotPoint'])> 0, 1, -1 )
df['Target30'] = np.where( (df['c1'] - df['PivotPoint'])> 0, 1, -1 )

df_raw = df.copy()

# Define the parameters to test
sma_periods = [5, 6, 7, 8, 9, 10, 12, 15]
rsi_periods = [9, 10, 14, 21, 28]
#buy_thresholds = [25, 30, 35, 40]
#sell_thresholds = [60, 65, 70, 80]

Result = []

# Create a list of all parameter combinations to test
parameters = list(product(sma_periods, rsi_periods))
for sma_prd, rsi_prd in parameters:

    df = df_raw.copy()
    # Compute technical indicators
    df['SMA'] = talib.SMA(df['Close'], timeperiod=sma_prd)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=rsi_prd)
    df['MFI'] = talib.MFI(df['High'],df['Low'],df['Close'],df['Volume'] )
    df.dropna(inplace=True)

    X = df[['SMA', 'RSI', 'MFI']]


    for i in range(1,31):
        tar_var = f"Target{i}"

        y = df[tar_var]
        #print(tar_var)
        result = get_accuracy(X,y)

        Result.append([i] + result + [sma_prd, rsi_prd])
        print(i)

    print(Result)
    result_df = pd.DataFrame(
        Result,
        columns=['target-var', 'in-sample-accuracy', 'out-of-sample-accuracy', 'sma', 'rsi']
    )

# store the result to the csv file
print(result.shape)

result_df.to_csv("../out/predictive_power2.csv")

print("finish running")

exit(0)

print(df[['Open', 'o1', 'Close', 'c1', 'Target', 'Target2']].head().to_string())

X = df[['SMA', 'RSI']]
y=df['Target']
print("target 1")
get_accuracy(X,y)


y=df['Target2']
print("target 2")
get_accuracy(X,y)

y=df['Target3']
print("target 3")
get_accuracy(X,y)

y=df['Target4']
print("target 4")
get_accuracy(X,y)

y=df['Target5']
print("target 5")
get_accuracy(X,y)

y=df['Target6']
print("target 6")
get_accuracy(X,y)
