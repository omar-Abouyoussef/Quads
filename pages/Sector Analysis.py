import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf
import datetime as dt
from tradingview_screener import Query, Column 

def get_market_info(market):
    market_info = (Query().select('country','name','exchange','market','sector','close', 'volume').
                   where(Column('volume') > 5000).
                set_markets(market).
                limit(20000).
                get_scanner_data())[1]
    if market == 'egypt':
        infot = market_info
        infor = pd.read_csv('egx_companies.csv')
        info = pd.concat([infot[['name','exchange']], infor.sector], axis=1, join='inner')
    else:
        info = market_info
    return info


def scale(x):
    x_scaled = (x - x.mean())/x.std()
    return x_scaled


#######
#inputs
#######
sectors = st.session_state.sectors

market = st.selectbox(label='Market:',
                       options = ['america','canada', 'uk' 'germany','uae', 'ksa', 'egypt'],
                       key='market')
market = st.session_state.market

cycle = st.selectbox(label='Cycle:',
                       options = ['Short-term', 'Medium-term', 'Long-term'],
                       key='cycle')
cycle = st.session_state.cycle

info = get_market_info(market=market)
# sector = 'market'
sector = st.selectbox(label='Sector:',
                       options = sectors.name if market=='america' else info.sector.unique().tolist() + ['Market'],
                       key='sector')
sector = st.session_state.sector


standardize = st.selectbox(label='Standardized:',
                       options = ['Yes', 'No'],
                       key='standardize')
standardize = st.session_state.standardize

sector_symbol = sectors[sectors['name']==sector]['symbol'] if market == 'america' else sector    

if cycle == 'Long-term':
    if standardize == 'Yes':
        st.write(st.session_state.df_50_100[st.session_state.df_50_100['Sector']==sector_symbol][cycle])
        series = scale(st.session_state.df_50_100[st.session_state.df_50_100['Sector']==sector_symbol][cycle])
        fig = px.line(series, line_shape="spline")
        fig.add_hline(1.27, line_width=1, line_dash="dash")
        fig.add_hline(-1.27, line_width=1, line_dash="dash")


    else:
        series = st.session_state.df_50_100[st.session_state.df_50_100['Sector']==sector_symbol][cycle]
        fig = px.line(series,line_shape="spline")


else:
    if standardize == 'Yes':
        series = scale(st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_symbol][cycle])
        fig = px.line(series,line_shape="spline")
        fig.add_hline(1.27, line_width=1, line_dash="dash")
        fig.add_hline(-1.27, line_width=1, line_dash="dash")


    else:
        series = st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_symbol][cycle]
        fig = px.line(series,line_shape="spline")

st.plotly_chart(fig)

################
#################
# if market == 'america':
#     us_stocks = pd.read_csv('us_stocks_cleaned.csv',keep_default_na=False)
#     stock_list = us_stocks[us_stocks.Sector==sector].reset_index(drop=True)['Symbol'].to_list()


#     etf = sectors[sectors.symbol==sector_symbol.values[0]]['etf'].values[0]
#     stock_list.append(etf)

#     if sector != 'Market':
#         end = dt.date.today()
#         start = end - dt.timedelta(365)
#         interval="1wk"
#         yf.pdr_override()
        
#         @st.cache_data
#         def get_data(stock_list, start, end, interval):
#             return pdr.get_data_yahoo(stock_list, start, end, interval)

#         data = get_data(stock_list, start, end, interval)
#         close_prices = data["Close"]
#         close_prices.dropna(inplace=True, axis=1)
#         returns = close_prices.diff().dropna()

#         corrcoef = returns.corr()[etf]
#         beta =  (returns.cov() / returns[etf].var())[etf]
#         df = pd.concat([beta.round(2), corrcoef.round(2)], axis =1)
#         df.columns = ['Beta', 'Correlation']

#         df = us_stocks.set_index('Symbol').join(df,how ='right')[['Name', 'Sector', 'Industry','Beta', 'Correlation']].fillna('ETF')
#         st.write(df)
#     else:
#         pass
