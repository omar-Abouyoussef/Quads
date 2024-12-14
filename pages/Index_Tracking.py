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
import statsmodels.api as sm
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

@st.cache_data
def get_us_close_prices(stock_list, start, end, interval='1d'):
    close = pdr.get_data_yahoo(stock_list, start, end, interval)
    return close

sectors = ['MM', 'SB', 'SE', 'SF', 'SI', 'SK', 'SL', 'SP', 'SS', 'SU', 'SV', 'SY']

sector_etf_dic = {'MM':'RSP', 'SB':'XLB', 'SL':'XLC', 'SF':'XLF', 'SI':'XLI', 'SK':'XLK', 'SP':'XLP', 'SS':'XLRE', 'SU':'XLU', 'SV':'XLV', 'SY':'XLY', 'SE':'XLE'}
sector_names_us_dic = {'Basic Materials': 'SB', 'Telecommunications': 'SL', 'Finance': 'SF', 'Industrials': 'SI', 'Technology': 'SK', 'Consumer Staples': 'SP', 'Real Estate': 'SS', 'Utilities': 'SU', 'Health Care': 'SV', 'Consumer Discretionary': 'SY', 'Energy': 'SE'}
# day_5_suffix = 'FD'
# day_20_suffix = 'TW'
# day_50_suffix = 'FI'
# day_100_suffix = 'OH'
# day_200_suffix = 'TH'

duration_dic={'Short-term':'TW','Medium-term':'FI'}
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
          max_value=30,
          value=5)

regularization = st.slider(label='Penalty:',
          min_value=0,
          max_value=10,
          value=3,
          help="""Penalty controls the regularization parameter in LASSO Regrssion, which helps in controlling
                the number of equities held in the portfolio. Higher values shrinks the regression coefficients (portoflio weights) towards zero
                allowing the portfolio to be more sparse, hence decreasing transactions costs but increasing Tracking error, and limits opportunities""")
rho = st.slider(label='rho:',
                  min_value=0.0,
                  max_value=1.0,
                  value=0.8,
                step=0.1,
help="ElasticNet rho parameter: controls l1 and l2 norm regularization. rho=1 applies l1 norm regularization- LASSO. rho=0 applies l2 regularization- Ridge regression. Rho and Penalty parameters both have implications on diversification")
##################
##################
#Fetch Index
# index = get_data(sector,duration,n,frequencies[0]) + 1


if market == 'america':
    index = st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_names_us_dic[sector_name]][duration]

    stocks=stocks[(stocks['Sector']==sector_name) & (stocks['Volume']>500000)]['Symbol']
    #stocks=stocks.loc[(stocks['Sector']==sector_name) & (stocks['Volume'].astype(int)>500000), ['Symbol']]
    stock_list = stocks.values.tolist()
    start = index.index[0]
    end = dt.date.today()
    yf.pdr_override()

    #Fetch CLose Prices
    close = get_us_close_prices(stock_list+[sector_etf_dic[sector_names_us_dic[sector_name]]]+['RINF'] + ['TLT'] + ['UVXY'],
                                start=start, end=end, interval = '1d')['Close']
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

###############################
##############################

df = pd.merge(index, close,
                        left_on=index.index,
                        right_on=close.index).set_index("key_0").astype(float)
df.index.name = "Date"
df = df.tail(1000)
n = (df.isna().sum()>30)
new_tickers = n[n==True].index.to_list()

deactivated_tickers = df.tail(1).isna().sum()>0
deactivated_tickers = deactivated_tickers[deactivated_tickers==True].index.to_list()

df = df.drop(new_tickers+deactivated_tickers, axis=1)

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



lowess = sm.nonparametric.lowess

smooth = pd.DataFrame(
    lowess(endog=reg_data['INDEX'], exog=reg_data['INDEX'].index, frac=0.03),
    index=df.index
                      )

#####################
#Model
#####################
coefs = []
intercept = []
score = []
date = []

X = reg_data.drop("INDEX", axis=1)
y=reg_data["INDEX"]

#X = X.shift(1)[1:]
# y=reg_data["INDEX"][1:]

############
#Smoothing

lowess = sm.nonparametric.lowess
smooth = pd.DataFrame(
    lowess(endog=reg_data['INDEX'], exog=reg_data['INDEX'].index, frac=0.03),
    index=df.index
                      )
smoothed = smooth.iloc[:,1]


window_size = 30
for i in range(0, len(X) - window_size + 1,  rebalance):

    # Extract the current rolling window
    X_window = X.iloc[i:i+window_size]
    y_window = y.iloc[i:i+window_size]
    model = linear_model.ElasticNet(alpha=regularization, l1_ratio=rho, positive=True)
    model.fit(X_window,y_window)

    coefs.append(model.coef_)
    intercept.append(model.intercept_)
    score.append(model.score(X_window,y_window))
    date.append(X_window.index[-1])
params = pd.DataFrame(coefs, index=date)

weights = params.apply(lambda x: abs(x)/abs(x).sum(), axis=1)

weights.index.name='Date'
params.columns = X.columns
weights.columns = X.columns

X_temp = X.loc[params.index,:]
fit = params.mul(X_temp).sum(axis=1)


f"\n\n\n Tracking error: {np.round((1-score[-1])*100,2)}%"
fig = go.Figure()
fig.add_trace(go.Scatter(y= y.loc[params.index,], x=params.index, name='Index', mode='lines'))
fig.add_trace(go.Scatter(y= fit + intercept, x=params.index, name='Tracker', mode='lines', line=dict(dash='dot')))
#fig.add_trace(go.Scatter(y=smoothed, x=params.index, name='Smoothed Index', mode='lines'))
fig.update_layout(title_text="Index Tracking", xaxis_title="", yaxis_title="")

st.plotly_chart(fig)

st.markdown(""" In the Sector Analysis page the graph offered valuable insight into the current sector sentiment and its cycle. One of the major benefits is that it is Mean-reverting.
\n\n The graph shows a portoflio that tracks the sector cycle/sentiment and the tracking error. \n\n Select the rebalancing frequency for the portoflio and the penalty term which controls the weights in the portoflio. \n\n ***more info?***  *check the question mark of the penalty option* """)

"""### Weights

"""

# fig  = px.line(params)
# fig.update_layout(title_text="Index Tracking Coefficients", xaxis_title="", yaxis_title="")
# fig.add_hline(y=0)
# st.plotly_chart(fig)

fig  = px.line(weights.round(2)*100)
fig.update_layout(title_text="Index Tracking Weights", xaxis_title="", yaxis_title="")
st.plotly_chart(fig)
#################
#Heatmap
##################

alltime_zero_weight_tickers= ((weights==0.0).sum()==weights.shape[0]).index.to_list()
alltime_zero_weight_tickers= (weights == 0.0).sum()==weights.shape[0]
alltime_zero_weight_tickers= alltime_zero_weight_tickers[alltime_zero_weight_tickers==True].index.to_list()
weights=weights.drop(alltime_zero_weight_tickers, axis=1)

col1, col2 = st.columns([0.8,0.1])
with col1:

    fig = go.Figure(
    data=go.Heatmap(
                     z=(weights.tail(252).T).astype(float), coloraxis="coloraxis",
        x=pd.to_datetime(weights.tail(252).index).date.astype(str), connectgaps=True, y=weights.columns, colorscale=[[0,'rgb(239,35,60)'],[1,'rgb(72,202,228)']],
        xgap=3, ygap=3
)
)
    fig.update_xaxes(type='category')
    fig.update_yaxes(type='category')
                   
    fig.update_layout(height = 600, title='Holdings Across Time')

    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig)

with col2:
    """###### Weights"""
    portfolio= (weights.iloc[-1,:].T)*100
    portfolio.name="Weights"
    portfolio.index.name="Tickers"
    st.write(portfolio.sort_values(ascending=False).round(2))
