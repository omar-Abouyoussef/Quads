
import datetime as dt
import pandas as pd
import requests
import streamlit as st
import io
from mito.streamlit.v1 import spreadsheet


@st.cache_data
def get_data(market:str, stock_list, start:dt.date, end:dt.date, key:str):
    
  
    close_prices = pd.DataFrame(columns=stock_list)
    for idx, ticker in enumerate(stock_list):
      try:
        url = f'https://eodhd.com/api/eod/{ticker}.{market}?from={start}&to={end}&filter=close&period=d&api_token={key}&fmt=json'
        res = requests.get(url)
        if res.status_code == 200:
          close = requests.get(url).json()
          close_prices[ticker] = close
      except:
        pass
    url = f'https://eodhd.com/api/eod/{stock_list[0]}.{market}?from={start}&to={end}&filter=date&period=d&api_token={key}&fmt=json'
    res = requests.get(url)
    if res.status_code == 200:
      date = res.json()
      close_prices['date'] = date
      close_prices.set_index('date', inplace=True)
      return close_prices



############
#streamlit


st.set_page_config(page_title="Download Data")
st.title('Download data')


##############################
#inputs
##########################

#inputs
country = st.selectbox(label='Country:',
                       options = ['Egypt', 'United States', 'Saudi Arabia'],
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




codes = {'Egypt':'EGX', 'United States':'US', 'Saudi Arabia':'SR'}


close_prices = get_data(market = codes[country], stock_list= tickers.split(" "),
                        start=start, end=end, key=st.secrets['eod_api_key'])

if close_prices is not None:
  close_prices, code = spreadsheet(close_prices)
  st.write(close_prices)

  st.download_button(
      label="Download",
      data=close_prices.to_csv(),
      file_name='data.csv',
  )
else:
  st.write('Ticker not available')
