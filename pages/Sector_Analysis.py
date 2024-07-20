import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf
import datetime as dt



def scale(x):
    x_scaled = (x - x.mean())/x.std()
    return x_scaled

sectors = st.session_state.sectors


#######
#inputs
#######

cycle = st.selectbox(label='Cycle:',
                       options = ['Short-term', 'Medium-term', 'Long-term'],
                       key='cycle')
cycle = st.session_state.cycle

sector = st.selectbox(label='Sector:',
                       options = sectors.name,
                       key='sector')
sector = st.session_state.sector



standardize = st.selectbox(label='Standardized:',
                       options = ['Yes', 'No'],
                       key='standardize')
standardize = st.session_state.standardize

sector_symbol = sectors[sectors['name']==sector]['symbol']


if cycle == 'Long-term':
    if standardize == 'Yes':
        series = scale(st.session_state.df_50_100[st.session_state.df_50_100['Sector']==sector_symbol.values[0]][cycle])
        fig = px.line(series)
        fig.add_hline(1.27, line_width=1, line_dash="dash")
        fig.add_hline(-1.27, line_width=1, line_dash="dash")


    else:
        series = st.session_state.df_50_100[st.session_state.df_50_100['Sector']==sector_symbol.values[0]][cycle]
        fig = px.line(series)


else:
    if standardize == 'Yes':
        series = scale(st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_symbol.values[0]][cycle])
        fig = px.line(series)
        fig.add_hline(1.27, line_width=1, line_dash="dash")
        fig.add_hline(-1.27, line_width=1, line_dash="dash")


    else:
        series = st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_symbol.values[0]][cycle]
        fig = px.line(series)

st.plotly_chart(fig)


us_stocks = pd.read_csv('us_stocks_cleaned.csv')
stock_list = us_stocks[us_stocks.Sector==sector].reset_index(drop=True)['Symbol'].to_list()


etf = sectors[sectors.symbol==sector_symbol.values[0]]['etf'].values[0]
stock_list.append(etf)

if sector != 'Market':
    end = dt.date.today()
    start = end - dt.timedelta(365)
    yf.pdr_override()
    data = pdr.get_data_yahoo(stock_list, start, end)

    close_prices = data["Close"]
    close_prices.dropna(inplace=True, axis=1)
    returns = close_prices.diff().dropna()

    corrcoef = returns.corr()[etf]
    beta =  (returns.cov() / returns[etf].var())[etf]
    df = pd.concat([beta.round(2), corrcoef.round(2)], axis =1)
    df.columns = ['Beta', 'Correlation']

    df = us_stocks.set_index('Symbol').join(df,how ='right')[['Name', 'Sector', 'Industry','Beta', 'Correlation']].fillna('ETF')
    st.write(df)
else:
    pass