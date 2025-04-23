import streamlit as st
from datetime import datetime as dt
import pandas as pd
import numpy as np
from egxpy.download import get_EGXdata, get_EGX_intraday_data, get_OHLCV_data
from egxpy.download import _get_intraday_close_price_data
import time

from concurrent.futures import ThreadPoolExecutor, as_completed



@st.cache_data
def eod_cache_func(tickers, interval, start, end, date):
  return get_EGXdata(tickers,interval,start,end)
  
# Footer
st.title('Download Data')

##############################
#inputs
##########################
#Tickers
tickers = st.text_input(label='Ticker(s): Enter all Caps',
                       key = 'tickers',
                       value='ABUK',
                      )
tickers = st.session_state.tickers.upper()

interval = st.selectbox(label='Interval',
                       options = ['Daily','Weekly','Monthly','1 Minute','5 Minute','30 Minute'],
                       key='interval',
                      )
interval = st.session_state.interval

start = st.date_input(label='Start date:',
              key='start')
start = st.session_state.start

end = st.date_input(label='End date:',
              key='end')
end = st.session_state.end

date = dt.today().date()

if start < end:
    start_time= time.time()
    if interval in ['1 Minute','5 Minute','30 Minute']:
            df = get_EGX_intraday_data(tickers.split(" "),interval,start,end)

    else:
        df = eod_cache_func(tickers.split(" "),interval,start,end,date)
        df.index = df.index.date
    end_time= time.time()
    st.write(df)
    st.write(f'{end_time-start_time:.2f} seconds')
    st.write(f"Samples:{df.shape[0]}")

# Download Button
    st.download_button(
        label="Download Data",
        data=df.to_csv(),
        file_name="Data.csv",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    

else:
    pass

st.write("Note: Intraday data is delayed by 20 minutes.")
st.markdown("<p class='footer'> &copy EGXLytics | 100% Free & Open Source</p>", unsafe_allow_html=True)

