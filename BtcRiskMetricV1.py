##########################################################
#          Last Update: 27 Nov 2021                      #
#   old code modified with new normalization             #
##########################################################  

import warnings
import pandas as pd
import numpy as np
import quandl as quandl
from datetime import date
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import date, timedelta
warnings.filterwarnings('ignore')

df = quandl.get("BCHAIN/MKPRU", api_key="FYzyusVT61Y4w65nFESX").reset_index()
btcdata = yf.download(tickers='BTC-USD', period="1d", interval="1m")["Close"]
df.loc[len(df)] = [date.today(), btcdata.iloc[-1]]

df = df[df["Value"] > 0]
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values(by="Date", inplace=True)
f_date = pd.to_datetime(date(2011, 8, 8))
E_date = pd.to_datetime(date(2009, 1, 3))  # genesis
delta = f_date - E_date
df = df[df.Date > f_date]
df.reset_index(inplace=True)

def normalization(data):
    normalized = (data - data.cummin()) / (data.cummax() - data.cummin())
    return normalized

def ossValue(days):
    X = np.array(np.log10(df.ind[:days])).reshape(-1, 1)
    y = np.array(np.log10(df.Value[:days]))
    reg = LinearRegression().fit(X, y)
    values = reg.predict(X)
    return values[-1]

############## 400MA ########################################
df['400MA'] = 0
for i in range(0, df.shape[0]):
    df['400MA'][i] = df['Value'][0 if i < 400 else (i - 400): i].dropna().mean()

df['400MArisk'] = 0
for i in range(0, df.shape[0]):
    df['400MArisk'][i] = (df['Value'][i] / df['400MA'][i])

############## Mayer Multiple ########################################
df['200MA'] = 0
for i in range(0, df.shape[0]):
    df['200MA'][i] = df['Value'][0 if i < 200 else (i - 200): i].dropna().mean()

df['Mayer'] = 0
for i in range(0, df.shape[0]):
    df['Mayer'][i] = (df['Value'][i] / df['200MA'][i])

############## Puell Multiple ########################################
df["btcIssuance"] = 7200 / 2 ** (np.floor(df["index"] / 1458))
df["usdIssuance"] = df["btcIssuance"] * df["Value"]

df['MAusdIssuance'] = 0
for i in range(0, df.shape[0]):
    df['MAusdIssuance'][i] = df['usdIssuance'][0 if i < 365 else (i - 365): i].dropna().mean()

df['PuellMultiple'] = 0
for i in range(0, df.shape[0]):
    df['PuellMultiple'][i] = df['usdIssuance'][i] / df['MAusdIssuance'][i]

############### Price/52W MA ########################################
df["365MA"] = 0
for i in range(0, df.shape[0]):
    df["365MA"][i] = df["Value"][0 if i < 365 else (i - 365): i].dropna().mean()

df["Price/52w"] = 0
for i in range(0, df.shape[0]):
    df["Price/52w"][i] = (df["Value"][i] / df["365MA"][i])

############## Sharpe Ratio ########################################
df["Return%"] = df["Value"].pct_change() * 100

df["365Return%MA-1"] = 0
for i in range(0, df.shape[0]):
    df["365Return%MA-1"][i] = df["Return%"][0 if i < 365 else (i - 365): i].dropna().mean() - 1

df["365Return%STD"] = 0
for i in range(0, df.shape[0]):
    df["365Return%STD"][i] = df["Return%"][0 if i < 365 else (i - 365): i].dropna().std()

df["Sharpe"] = 0
for i in range(0, df.shape[0]):
    df["Sharpe"][i] = (df["365Return%MA-1"][i] / df["365Return%STD"][i])

############# Power Law ###########################################
df["ind"] = [x + delta.days for x in range(len(df))]
df["PowerLaw"] = np.log10(df.Value) - [ossValue(x + 1) for x in range(len(df))]

#################### avg ################################
indicators = ["PuellMultiple", "Price/52w", "PowerLaw", "Sharpe", "Mayer","400MArisk"]

for item in indicators:
    df[item].update(normalization(df[item]))

df["avg"] = df[['PuellMultiple', 'PowerLaw', 'Sharpe', '400MArisk', "Mayer"]].mean(axis=1)
df["avg"] = df["avg"] * df.index**0.395
df["avg"] = (df["avg"] - df["avg"].cummin()) / (df["avg"].cummax() - df["avg"].cummin())
df = df[df.index > 800]

#################### Plot ################################
fig = make_subplots(specs=[[{"secondary_y": True}]])

xaxis = df.Date

fig.add_trace(go.Scatter(x=xaxis, y=df.Value, name="Price", line=dict(color="gold")), secondary_y=False)
fig.add_trace(go.Scatter(x=xaxis, y=df["avg"], name="Risk", line=dict(color="white")), secondary_y=True)

AnnotationText = f"****Last BTC price: {round(df['Value'].iloc[-1])}**** Risk: {round(df['avg'].iloc[-1],2)}****"
fig.add_hrect(y0=0.5, y1=0.4, line_width=0, fillcolor="green", opacity=0.2, secondary_y=True)
fig.add_hrect(y0=0.4, y1=0.3, line_width=0, fillcolor="green", opacity=0.3, secondary_y=True)
fig.add_hrect(y0=0.3, y1=0.2, line_width=0, fillcolor="green", opacity=0.4, secondary_y=True)
fig.add_hrect(y0=0.2, y1=0.1, line_width=0, fillcolor="green", opacity=0.5, secondary_y=True)
fig.add_hrect(y0=0.1, y1=0.0, line_width=0, fillcolor="green", opacity=0.6, secondary_y=True)
fig.add_hrect(y0=0.6, y1=0.7, line_width=0, fillcolor="red", opacity=0.3, secondary_y=True)
fig.add_hrect(y0=0.7, y1=0.8, line_width=0, fillcolor="red", opacity=0.4, secondary_y=True)
fig.add_hrect(y0=0.8, y1=0.9, line_width=0, fillcolor="red", opacity=0.5, secondary_y=True)
fig.add_hrect(y0=0.9, y1=1.0, line_width=0, fillcolor="red", opacity=0.6, secondary_y=True)
fig.update_xaxes(title="Date")
fig.update_yaxes(title="Price", type='log', showgrid=False)
fig.update_yaxes(title="Risk", type='linear', showgrid=True, tickmode='array', tick0=0.0, dtick=0.1, secondary_y=True)
fig.update_layout(template="plotly_dark", title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
fig.show()
