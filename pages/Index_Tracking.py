#import the libraries
import datetime as dt
import yfinance as yf
import pandas_datareader.data as pdr
import plotly.graph_objects as go
import plotly.express as px
import plotly.tools as tls
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tradingview_screener import Query, Column, get_all_symbols
from tvDatafeed import TvDatafeed, Interval
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import linear_model
from retry import retry
import streamlit as st

@retry((Exception), tries=10, delay=1, backoff=0)
def get_data(sector, suffix,n,freq):
    tv = TvDatafeed()
    response = tv.get_hist(symbol=f'{sector}{suffix}',
                    exchange='INDEX',interval=freq,
                    n_bars=n)['close']
    return response



sectors = ['MM', 'SB', 'SE', 'SF', 'SI', 'SK', 'SL', 'SP', 'SS', 'SU', 'SV', 'SY']

sector_etf_dic = {'MM':'RSP', 'SB':'XLB', 'SL':'XLC', 'SF':'XLF', 'SI':'XLI', 'SK':'XLK', 'SP':'XLP', 'SS':'XLRE', 'SU':'XLU', 'SV':'XLV', 'SY':'XLY'}
sector_names_us_dic = {'Basic Materials': 'SB', 'Telecommunications': 'SL', 'Finance': 'SF', 'Industrials': 'SI', 'Technology': 'SK', 'Consumer Staples': 'SP', 'Real Estate': 'SS', 'Utilities': 'SU', 'Health Care': 'SV', 'Consumer Discretionary': 'SY', 'Energy': 'SE'}
# day_5_suffix = 'FD'
# day_20_suffix = 'TW'
# day_50_suffix = 'FI'
# day_100_suffix = 'OH'
# day_200_suffix = 'TH'

duration_dic={'Short-term':'TW','Medium-term':'FI', 'Long-term':'OH'}
markets=['america','egypt']

in_monthly = Interval.in_monthly
in_weekly = Interval.in_weekly
in_daily = Interval.in_daily
frequencies = [in_daily, in_weekly, in_monthly]


market = st.selectbox(label='Market:',
                       options = markets,
                       key='market')
market = st.session_state.market

if market == 'america':
    stocks = pd.read_csv('us_stocks_cleaned.csv', header=0)
    sector_names_list = stocks['Sector'].unique()

elif market == 'egypt':
    stocks = pd.read_csv('egx_companies.csv', header=0)
    sector_names_list = stocks['sector'].unique()


sector_name = st.selectbox(label='Sector:',
                       options = sector_names_list,
                       key='sector_name')
sector_name = st.session_state.sector_name

duration = st.selectbox(label='Duration:',
                       options = list(duration_dic.keys()),
                       key='duration')
duration = st.session_state.duration

rebalance = st.slider(label='Rebalance every days:',
          min_value=1,
          max_value=252,
          value=5)

regularization = st.slider(label='Penalty:',
          min_value=0,
          max_value=100,
          value=1,
          help="""Penalty controls the regularization parameter in LASSO Regrssion, which helps in controlling
                the number of equities held in the portfolio. Higher values shrinks the regression coefficients (portoflio weights) towards zero
                alloing the portfolio to be more sparse, hence decreasing transactions costs but increasing Tracking error.""")

##################
##################
#Fetch Index
# index = get_data(sector,duration,n,frequencies[0]) + 1


if market == 'america':
    index = st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_names_us_dic[sector_name]][duration]

    stocks=stocks[stocks['Sector']==sector_name]['Symbol']
    stock_list = stocks.values.tolist()

    start = index.index[0]
    end = dt.date.today()
    yf.pdr_override()

    #Fetch CLose Prices
    close = pdr.get_data_yahoo(stock_list+[sector_etf_dic[sector_names_us_dic[sector_name]]]+['RINF'] + ['TLT'] + ['UVXY'], interval = '1d', start=start, end=end)['Close']
    
elif market == 'egypt':
    index = st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_name][duration]

    stocks=stocks[stocks['sector']==sector_name]['name']
    stock_list = stocks.values.tolist()


    close = st.session_state.close_price_data
    close.index = pd.to_datetime(close.index.date)
    close.index.name = 'Date'

    stock_list = list(set(stock_list).intersection(close.columns.values.tolist()))

    close = close.loc[:,stock_list]

index.name = 'INDEX'
index.index = pd.to_datetime(pd.to_datetime(index.index).date)
index.index.name = 'Date'
# print(index)
###############################
##############################

df = pd.merge(index, close,
                        left_on=index.index,
                        right_on=close.index).set_index("key_0").astype(float)
df.index.name = "Date"
df = df.tail(1000)
n = (df.isna().sum()>60)
bad_tickers = n[n==True].index.to_list()

df = df.drop(bad_tickers, axis=1)
df = df.dropna()



#"LASSO"
# reg_data= df
# X = reg_data.drop("INDEX", axis=1)
# y=reg_data["INDEX"]

# model = linear_model.ElasticNet(alpha=1, l1_ratio=1, positive=False)
# model.fit(X, y)
# print(model.coef_.round(2))
# print(model.intercept_)
# model.score(X, y)

# fig = go.Figure()
# fig.add_trace(go.Scatter(y= y, x=y.index, name='original', mode='lines'))
# fig.add_trace(go.Scatter(y= model.predict(X), x=y.index, name='predicted', mode='lines'))
# fig.show()

# """ # Rolling Lasso"""



reg_data = df

X = reg_data.drop("INDEX", axis=1)
y=reg_data["INDEX"]

coefs = []
intercept = []
score = []
date = []
x = reg_data.drop("INDEX", axis=1)
window_size = 30

for i in range(0, len(x) - window_size + 1,  rebalance):

    # Extract the current rolling window
    X_window = x.iloc[i:i+window_size]
    y_window = reg_data["INDEX"].iloc[i:i+window_size]
    model = linear_model.ElasticNet(alpha=1, l1_ratio=1, positive=True)
    model.fit(X_window,y_window)

    coefs.append(model.coef_)
    intercept.append(model.intercept_)
    score.append(model.score(X_window,y_window))
    date.append(X_window.index[-1])
params = pd.DataFrame(coefs, index=date)
weights_coin = params.apply(lambda x: abs(x)/abs(x).sum(), axis=1)
params.columns = X.columns

X_temp = X.loc[params.index,:]
fit = params.mul(X_temp).sum(axis=1)


fig = go.Figure()
fig.add_trace(go.Scatter(y= y.loc[params.index,], x=params.index, name='original', mode='lines'))
fig.add_trace(go.Scatter(y= fit + intercept, x=params.index, name='predicted', mode='lines'))
fig.update_layout(title_text="Index Tracking", xaxis_title="", yaxis_title="")

st.plotly_chart(fig)


"""# Weights

"""

# fig  = px.line(params)
# fig.update_layout(title_text="Index Tracking Coefficients", xaxis_title="", yaxis_title="")
# fig.add_hline(y=0)
# st.plotly_chart(fig)


weights = params.apply(lambda x: abs(x)/abs(x).sum(), axis=1)
fig  = px.line(weights.round(2)*100)
fig.update_layout(title_text="Index Tracking Weights", xaxis_title="", yaxis_title="")
st.plotly_chart(fig)
#################
#Heatmap
##################
plt.figure(figsize=(14,8))



st.write(weights.iloc[-1,:].T)
fig = go.Figure(
    data=go.Heatmap(
                     z=(params.T.iloc[:,-10:]>0).astype(int), coloraxis="coloraxis",
        x=params.index, y=params.columns, colorscale=[[0,'rgb(239,35,60)'],[1,'rgb(72,202,228)']],
        xgap=4,ygap=4
)
)

fig.update_coloraxes(showscale=False)
st.plotly_chart(fig)
# sns.heatmap(params.T.iloc[:,-30:]>0)

















