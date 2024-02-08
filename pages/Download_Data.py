import datetime as dt
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf
import requests
import streamlit as st
import io

@st.cache_data
def get_data(market:str, stock_list:list, start:dt.date, end:dt.date, key:str):
    
    close_prices = pd.DataFrame(columns=stock_list)
    if market == "US":
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
        fx_list = []
        for stock in stock_list:
            fx_list.append(stock+"=X")
        return pdr.get_data_yahoo(fx_list, start, end)["Close"]
    
    else:
        ticker_list = []
        for stock in stock_list:
            ticker_list.append(stock+f'.{market}')
        return pdr.get_data_yahoo(ticker_list, start, end)["Close"]



############
#streamlit


st.set_page_config(page_title="Download Data")
st.title('Download data')


##############################
#inputs
##########################

#inputs
country = st.selectbox(label='Country:',
                       options = ['Egypt', 'United States', 'Saudi Arabia', 'Forex'],
                       key='country')
country = st.session_state.country


#Tickers
tickers = st.text_input(label='Ticker(s):',
                       key = 'tickers',
                       value='ABUK',
                      )
tickers = st.session_state.tickers.upper()

#####
start = st.date_input(label='Start date:',
              key='start')
start = st.session_state.start

end = st.date_input(label='End date:',
              key='end')
end = st.session_state.end




codes = {'Egypt':'EGX', 'United States':'US', 'Saudi Arabia':'SR', 'Forex':'FOREX'}

yf.pdr_override()
close_prices = get_data(market = codes[country], stock_list= tickers.split(" "),
                        start=start, end=end, key=st.secrets['eod_api_key'])

if close_prices is not None:
  st.dataframe(close_prices)

  st.download_button(
      label="Download",
      data=close_prices.to_csv(),
      file_name='data.csv',
  )
else:
  st.write('Ticker not available')
