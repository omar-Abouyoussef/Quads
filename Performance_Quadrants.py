import datetime as dt
import pandas as pd
import requests
import pandas_datareader.data as pdr
import yfinance as yf
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
import investpy
import streamlit as st


def change(data, freq):
    return data.iloc[-1]/data.iloc[-(freq+1)] - 1

   
@st.cache_data
def get_data(market:str, stock_list:list, start:dt.date, end:dt.date, key:str):

    stock_list.sort()
    close_prices = pd.DataFrame(columns=stock_list)
    if market == "US":
         yf.pdr_override()
         return pdr.get_data_yahoo(stock_list, start, end)["Close"]

    elif market == "EGX":
         for idx, ticker in enumerate(stock_list):
           try:
             url = f'https://eodhd.com/api/eod/{ticker}.{market}?from={start}&to={end}&filter=close&period=d&api_token={key}&fmt=json'
             close = requests.get(url).json()
             close_prices[ticker] = close
           except:
             pass
         url = f'https://eodhd.com/api/eod/{stock_list[0]}.{market}?from={start}&to={end}&filter=date&period=d&api_token={key}&fmt=json'
         date = requests.get(url).json()
         close_prices['date'] = date
         close_prices.set_index('date', inplace=True)
         return close_prices

    elif market == 'FOREX':
         return pdr.get_data_yahoo(stock_list, start, end)["Close"]
    
    else:
         ticker_list = []
         for stock in stock_list:
          ticker_list.append(stock+f'.{market}')
         return pdr.get_data_yahoo(ticker_list, start, end)["Close"]
######################
####################

st.set_page_config(page_title="Performance Quadrant", layout="wide")
st.title('Performance Quadrant')
st.sidebar.header('Home')



#####
#Global Vars
#####
today = dt.date.today()
start = today - dt.timedelta(365)


codes = {'Egypt':'EGX', 'United States':'US', 'Saudi Arabia':'SR', 'Forex':'FOREX', 'Crypto':'CC'}

fx_list = ['EURUSD=X','JPY=X',
           'GBPUSD=X', 'AUDUSD=X',
           'NZDUSD=X', 'EURJPY=X',
           'GBPJPY=X', 'EURGBP=X',
           'EURCAD=X', 'EURSEK=X',
           'EURCHF=X', 'EURHUF=X',
           'EURJPY=X', 'CNY=X',
           'HKD=X', 'SGD=X',
           'INR=X', 'MXN=X',
           'PHP=X', 'IDR=X',
           'THB=X', 'MYR=X',
           'ZAR=X', 'RUB=X']

###############################
#inputs
country = st.selectbox(label='Country:',
                       options = ['Egypt', 'United States', 'Saudi Arabia', 'Forex'],
                       key='country')
country = st.session_state.country

plot = st.selectbox(label='Plot type:',
                    options=['Short-term|Medium-term', 'Short-term|Long-term', 'Medium-term|Long-term'],
                    key='plot')
plot = st.session_state.plot
##################################

if country == 'Forex':
    close_prices = get_data(market = codes[country], stock_list=fx_list,
                            start=start, end=today, key=st.secrets["eod_api_key"])
else:
    stock_list = investpy.stocks.get_stocks_list(country = country)
    close_prices = get_data(market = codes[country], stock_list=stock_list,
                            start=start, end=today, key=st.secrets["eod_api_key"])
    

    

close_prices.dropna(axis = 1, inplace = True)    

one_day_return = change(close_prices, 1)
two_day_return = change(close_prices, 2)
three_day_return = change(close_prices, 3)
weekly_return = change(close_prices, 5)
three_week_return = change(close_prices, 15)
one_month_return = change(close_prices, 22)
three_month_return = change(close_prices, 66)
six_month_return = change(close_prices, 132)


performance = pd.DataFrame(
list(
    zip(one_day_return,
        two_day_return,
        three_day_return,
        weekly_return,
        three_week_return,
        one_month_return,
        three_month_return,
        six_month_return)
        ),
        columns = ['1-Day', '2-Day', '3-Day', '1-Week', '3-Week', '1-Month', '3-Month', '6-Month'],
        index = close_prices.columns
        )

performance.dropna(inplace=True)
scaler = StandardScaler()
performance_scaled = scaler.fit_transform(performance)
fa = FactorAnalysis(1)
short_term= fa.fit_transform(performance_scaled[:,0:3]).reshape(-1)
medium_term=fa.fit_transform(performance_scaled[:,3:6]).reshape(-1)
long_term=fa.fit_transform(performance_scaled[:,-3:-1]).reshape(-1)
factors=pd.DataFrame(data = {"Short-term":short_term,
                               "Medium-term":medium_term,
                               "Long-term":long_term},
                       index = performance.index)

model=KMeans(n_clusters=4,random_state=0).fit(factors)
factors['Cluster']=model.labels_
#factors['Cluster']=factors['Cluster'].map({0:'Weakening',1:'Falling',2:'Improving',3:'Momentum'})

if plot == 'Short-term|Medium-term':
    fig=px.scatter(factors,x='Medium-term',y='Short-term',
                   hover_data=[factors.index],color=factors["Cluster"].astype(str))
    
elif plot == 'Medium-term|Long-term':
    fig = px.scatter(factors, x='Long-term', y='Medium-term',
                     hover_data = [factors.index],color=factors["Cluster"].astype(str))
else: 
   fig = px.scatter(factors, x='Long-term', y='Short-term',
                    hover_data = [factors.index],color=factors["Cluster"].astype(str))

fig.add_hline(y=0)
fig.add_vline(x=0)

st.plotly_chart(fig)

st.write("Top-right Quadrant: Siginfies extremely bullish and violent movement in price- suited for momentum plays. \n  Bottom-right: After a bullish move equties weakened and price started to drop. \n  \n  Bottom-Left Quadrant: Falling equties. \n  Top-left: Falling stocks started to improve their preformance attracting more buyers. \n\n\n  Note: Factor Scores are normally distributed with mean of zero. Scores assigned to stocks are, in effect, done on a relative basis. A stock in the bottom left quadrants does not always mean that it is falling instead it is underperforming the rest. \n  Example: If the entire market is extremely bullish, and almost most stocks are uptrending, equties lying in the bottom left quadrant are underperformers but still bullish.")


