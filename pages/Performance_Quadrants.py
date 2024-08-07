import datetime as dt
import pandas as pd
import numpy as np
import requests
import pandas_datareader.data as pdr
import yfinance as yf
import plotly.express as px
from factor_analyzer import FactorAnalyzer, ConfirmatoryFactorAnalyzer, ModelSpecificationParser
from sklearn.cluster import KMeans
import investpy
import streamlit as st
import time
from concurrent.futures import ThreadPoolExecutor


def change(data, freq):
    return data.iloc[-1]/data.iloc[-(freq+1)] - 1


def download(ticker, market, start, end, key):
    close = None
    try:
      url = f'https://eodhd.com/api/eod/{ticker}.{market}?from={start}&to={end}&filter=close&period=d&api_token={key}&fmt=json'
      res = requests.get(url)
      if res.status_code == 200:
        close = requests.get(url).json()
    except:
      pass
    return ticker, close

def get_us_forex_data(stock, start, end):
    return pdr.get_data_yahoo([stock], start, end)["Close"]
    

@st.cache_data
def get_data(market:str, stock_list:list, start:dt.date, end:dt.date, key:str):
    
    
    if market in ["US", 'FOREX']:
        return pdr.get_data_yahoo(stock_list, start, end)["Close"]

    elif market == "EGX":
        close_prices = pd.DataFrame(columns=stock_list)
         
        markets = [market]*len(stock_list)
        starts = [start]*len(stock_list)
        ends = [end]*len(stock_list)
        keys = [key]*len(stock_list)

        s = time.perf_counter()
        with ThreadPoolExecutor(max_workers=12) as executor:
            for ticker, close in executor.map(download,stock_list, markets, starts, ends, keys):
                try:
                    close_prices[ticker] = close
                except:
                    pass
        url = f'https://eodhd.com/api/eod/{stock_list[0]}.{market}?from={start}&to={end}&filter=date&period=d&api_token={key}&fmt=json'
        res = requests.get(url)
        if res.status_code == 200:
            date = res.json()
            close_prices['date'] = date
            close_prices.set_index('date', inplace=True)
            e = time.perf_counter()
            st.write(f"Finished in {e-s:.4} s")
            return close_prices

    #     #####################################################
    #     # s = time.perf_counter()
    #     # for idx, ticker in enumerate(stock_list):
    #     #     try:
    #     #         url = f'https://eodhd.com/api/eod/{ticker}.{market}?from={start}&to={end}&filter=close&period=d&api_token={key}&fmt=json'
    #     #         close = requests.get(url).json()
    #     #         close_prices[ticker] = close
    #     #     except:
    #     #         pass
    #     # url = f'https://eodhd.com/api/eod/{stock_list[0]}.{market}?from={start}&to={end}&filter=date&period=d&api_token={key}&fmt=json'
    #     # date = requests.get(url).json()
    #     # close_prices['date'] = date
    #     # close_prices.set_index('date', inplace=True)
    #     # e = time.perf_counter()
    #     # st.write(f"Finished in {e-s:.4} s")
    #     # return close_prices
    #     #####################################################
    
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
           'EURCAD=X', 'EURJPY=X',
           'EURAUD=X', 'NZDJPY=X',
           'HKD=X', 'SGD=X',
           'INR=X', 'MXN=X',
           'PHP=X', 'IDR=X',
           'THB=X', 'MYR=X',
           'ZAR=X', 'RUB=X']

###############################
#inputs
############3
country = st.selectbox(label='Country:',
                       options = ['Egypt', 'United States', 'Saudi Arabia', 'Forex'],
                       key='country')
country = st.session_state.country

plot = st.selectbox(label='Plot type:',
                    options=['Short-term|Medium-term', 'Short-term|Long-term', 'Medium-term|Long-term'],
                    key='plot')
plot = st.session_state.plot

###########################


##################################
#download data
###########################

yf.pdr_override()
if country == 'Forex':
    close_prices = get_data(market = codes[country], stock_list=fx_list,
                            start=start, end=today, key=st.secrets["eod_api_key"])
    

    
elif country == 'United States':
    us_companies_info = pd.read_csv('companies.csv')
    etfs = us_companies_info[us_companies_info['ETF']=='Yes']['Ticker'].to_list()
    stock_list = investpy.stocks.get_stocks_list(country = country)
    close_prices = get_data(market = codes[country], stock_list=stock_list+etfs,
                            start=start, end=today, key=st.secrets["eod_api_key"])
    

else:
    stock_list = investpy.stocks.get_stocks_list(country = country)
    close_prices = get_data(market = codes[country], stock_list=stock_list,
                            start=start, end=today, key=st.secrets["eod_api_key"])
    
##################################

if country == "United States":
    price = st.number_input(label='Minimum price: ',
                              key='fltr')
    price = st.session_state.fltr

    if price:
        cols = close_prices.columns[close_prices.iloc[-1,:] > price]
        close_prices = close_prices[cols]

close_prices.dropna(axis = 1, inplace = True)    

one_day_return = change(close_prices, 1)
two_day_return = change(close_prices, 2)
three_day_return = change(close_prices, 3)
weekly_return = change(close_prices, 5)
two_week_return = change(close_prices, 10)
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
        two_week_return,
        three_week_return,
        one_month_return,
        three_month_return,
        six_month_return)
        ),
        columns = ['1-Day', '2-Day', '3-Day', '1-Week', '2-Week', '3-Week', '1-Month', '3-Month', '6-Month'],
        index = close_prices.columns
        )

performance.dropna(inplace=True)
st.session_state.performance = performance   


cfa = FactorAnalyzer(3, rotation = 'varimax').fit(performance.values)


loadings = pd.DataFrame(cfa.loadings_, index = performance.columns)
st.session_state.loadings = loadings

for idx, col in enumerate(loadings.columns):
    vars = loadings[loadings[col]>0.5].index

    if (len(set(vars) & set(["1-Day", "2-Day", "3-Day"])) >= 2):
       loadings.rename(columns={ loadings.columns[idx]: "Short-term" }, inplace = True)
    elif (len(set(vars) & set(["1-Week", "2-Week", "3-Week"])) >= 2):
       loadings.rename(columns={ loadings.columns[idx]: "Medium-term" }, inplace = True)
    elif (len(set(vars) & set(["1-Month", "3-Month", "6-Month"])) >= 2):
       loadings.rename(columns={ loadings.columns[idx]: "Long-term" }, inplace = True) 
print(loadings)


factors = pd.DataFrame(cfa.transform(performance.values),
                       index = performance.index,
                       columns = st.session_state.loadings.columns)
st.session_state.cfa = cfa


model=KMeans(n_clusters=4,random_state=0).fit(factors)
factors['Cluster']=model.labels_


#####
#input
######
tickers = st.text_input(label='Ticker(s)',
                        value=" ".join(factors.index.to_list()),
                        key='tickers',
                        help="Enter all uppercase!",
                        placeholder='Choose ticker(s)')

if tickers == "":
    st.write("Reload page to get all tickers")
else:
    if tickers.isupper():
        tickers = st.session_state.tickers.split(" ")
    else:
        st.error("Enter ticker(s) in Uppercase!")


#######   
if country == 'United States':

    factors = us_companies_info[['Ticker', 'Name', 'Sector', 'Industry']].join(factors, on = 'Ticker', how = 'right').set_index('Ticker')

elif country == 'Egypt':
    egx_companies_info = pd.read_csv('egx_companies.csv')
    factors = egx_companies_info[['Ticker', 'Name', 'Sector']].join(factors, on = 'Ticker', how = 'right').set_index('Ticker')
else:
    factors["Sector"] = "NA"

try:
    if plot == 'Short-term|Medium-term':
        fig=px.scatter(factors.loc[tickers,:],x='Medium-term',y='Short-term',
                       hover_data=[factors.loc[tickers,:].index], color=factors.loc[tickers,"Sector"].astype(str))
    
    elif plot == 'Medium-term|Long-term':
        fig=px.scatter(factors.loc[tickers,:],x='Long-term',y='Medium-term',
                       hover_data=[factors.loc[tickers,:].index], color=factors.loc[tickers,"Sector"].astype(str))
    else:
        fig=px.scatter(factors.loc[tickers,:],x='Long-term',y='Short-term',
                       hover_data=[factors.loc[tickers,:].index], color=factors.loc[tickers,"Sector"].astype(str))

    fig.add_hline(y=0)
    fig.add_vline(x=0)

    container = st.container()
    with container:
        plot, df = st.columns([0.7, 0.3])
        
        with plot:
            st.plotly_chart(fig)
            st.markdown(f"*Last available data point as of {close_prices.index[-1]}*\n  \n")
        with df:
            st.dataframe(factors)            


except:
    st.warning("Invalid ticker(s)")

st.markdown('''\n  \n  **Top-right Quadrant:** Siginfies extremely bullish and violent movement in price- suited for momentum plays. \n  \n  **Bottom-right:** After a bullish move equties weakened and price started to drop. \n  \n  **Bottom-Left Quadrant:** Falling equties. \n \n  **Top-left:** Falling stocks started to improve their preformance attracting more buyers. \n\n\n  **Note:**   \n\n\n   1. The absolute value of the scores signifies strength of the movement.   \n\n\n   2. Factor Scores are normally distributed with mean of zero. Scores assigned to stocks are, in effect, done on a relative basis. A stock in the bottom left quadrants does not always mean that it is falling instead it is underperforming the rest. \n  Example: If the entire market is extremely bullish, and almost most stocks are uptrending, equties lying in the bottom left quadrant are underperformers but still bullish. (rare case) \n\n\n''')
st.write("Check Model diagnostics before using")
