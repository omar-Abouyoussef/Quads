import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf
import datetime as dt
from tradingview_screener import Query, Column 
import statsmodels.api as sm
from scipy.signal import savgol_filter



def get_market_info(market):
    market_info = (Query().select('name','exchange','sector', 'volume',
                                      'return_on_equity_fq', 'return_on_invested_capital_fq', 'price_book_fq','return_on_equity', 'return_on_invested_capital',
                                      'price_earnings_current','close', 'price_target_low', 'price_target_average','price_target_high', 'market_cap_basic').
                   where(Column('volume') > 5000).
                set_markets(market).
                limit(20000).
                get_scanner_data())[1]

    if market == 'america':
        us_stock_data = pd.read_csv('us_stocks_cleaned.csv', index_col=0)
        ticker_GICS = us_stock_data['Sector'] 
        market_info = pd.merge(left=market_info, right=ticker_GICS, left_on=market_info.name, right_on=ticker_GICS.index, how='right').drop(['sector', 'key_0'], axis=1)
    
    if market == 'egypt':
        infot = market_info[['name','exchange','close', 'volume','market_cap_basic']]
        infor = pd.read_csv('egx_companies.csv')
        #info = pd.concat(infot[['name','exchange','close','volume','market_cap_basic']], infor.sector], axis=1, join='inner')
        info = pd.merge(left=infot[['name','exchange','close','volume','market_cap_basic']], right=infor, on='name')
        
        info = info[['name','sector','close','volume', 'market_cap_basic']]
    else:
        info = market_info
    return info


def US_fundamentals(country, sector, x,y):
    market_data = get_market_info(country)
    roe_pb = market_data[market_data.Sector==sector].dropna(subset='ticker').reset_index(drop=True)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=roe_pb[x],y=roe_pb[y], mode='markers', hovertext=roe_pb.name)
        )
    fig.update_layout(title=sector,xaxis_title = x, yaxis_title=y)
    return roe_pb, fig

def scale(x):
    x_scaled = (x - x.mean())/x.std()
    return x_scaled
    
# def denoise(x, period):
#     """smoothes a given time series using a convolution window

#     Args:
#         x (pandas series): A given time series

#     Returns:
#         decomposition.trend: smoothed trend series
#         decomposition.resid: residual 
#     """
#     decomposition=sm.tsa.seasonal_decompose(x,model="additive", period=period,two_sided=True,extrapolate_trend=1)
#     return decomposition.trend

def denoise(df, window=30, order=3):
    """smoothes a given time series using a savgol filter

    Args:
        x (pandas series): A given time series

    Returns:
            y_smooth (pandas series): The smoothed series 
    """
    # Apply Savitzky-Golay filter
    y_smooth = savgol_filter(df, window_length=window, polyorder=order)
    y_smooth = savgol_filter(df, window_length=30, polyorder=1)


    return y_smooth


us_sectors = pd.read_excel('sectors.xlsx', sheet_name='Sheet1')
#######
#inputs
#######

market = st.selectbox(label='Market:',
                       options = ['america','uae', 'ksa', 'egypt'],
                       key='market')
market = st.session_state.market

cycle = st.selectbox(label='Cycle:',
                       options = ['Short-term', 'Medium-term', 'Long-term'],
                       key='cycle')
cycle = st.session_state.cycle

info = get_market_info(market=market)
sector = st.selectbox(label='Sector:',
                       options = us_sectors.name if market=='america' else info.sector.unique().tolist() + ['Market'],
                       key='sector')
sector = st.session_state.sector


standardize = st.selectbox(label='Standardized:',
                       options = ('No', 'Yes'),
                       key='standardize')
standardize = st.session_state.standardize

if market != 'america':
    st.download_button(label='Download Market Price Data:',
                   data=st.session_state.close_price_data.to_csv(),
                      file_name='Market_data.csv')
                   
sector_symbol = us_sectors[us_sectors['name']==sector]['symbol'].values[0] if market == 'america' else sector    

if cycle == 'Long-term':
    if standardize == "Yes":
     
        series = st.session_state.df_50_100[st.session_state.df_50_100['Sector']==sector_symbol][cycle]
        zscore = (series - series.rolling(window=60).mean()) / series.rolling(window=60).std()
        
        fig = px.line(zscore, line_shape="spline")
        fig.add_hline(1.27, line_width=1, line_dash="dash")
        fig.add_hline(-1.27, line_width=1, line_dash="dash")


    else:
        series = st.session_state.df_50_100[st.session_state.df_50_100['Sector']==sector_symbol][cycle]
        smoothed_series = denoise(series,window=5,order=3)
        
        #fig = px.line(series,line_shape="spline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=smoothed_series, x=series.index, name="Smoothed"))
        fig.add_trace(go.Scatter(y=series, x=series.index, name="Actual"))
        
        
else:
        
    if standardize == 'Yes':
        series = st.session_state.df_20_50_smoothed[st.session_state.df_20_50_smoothed['Sector']==sector_symbol][cycle]
        smoothed_zscore = (series - series.rolling(window=60).mean()) / series.rolling(window=60).std()

        series = st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_symbol][cycle]
        zscore = (series - series.rolling(window=60).mean()) / series.rolling(window=60).std()
        
        # fig = px.line(zscore,line_shape="spline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=smoothed_zscore,
                         x=st.session_state.df_20_50_smoothed[st.session_state.df_20_50_smoothed['Sector']==sector_symbol][cycle].index, name='Smoothed Standardized'))

        fig.add_trace(go.Scatter(y=zscore,
                         x=zscore.index, name='Actual Standardized'))
        

        fig.add_hline(1.27, line_width=1, line_dash="dash")
        fig.add_hline(-1.27, line_width=1, line_dash="dash")


    else:
        series = st.session_state.df_20_50_smoothed[st.session_state.df_20_50_smoothed['Sector']==sector_symbol][cycle]
        
        # fig = px.line(series,line_shape="spline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state.df_20_50_smoothed[st.session_state.df_20_50_smoothed['Sector']==sector_symbol][cycle],
                         x=st.session_state.df_20_50_smoothed[st.session_state.df_20_50_smoothed['Sector']==sector_symbol][cycle].index, name='Smoothed'))

        fig.add_trace(go.Scatter(y=st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_symbol][cycle],
                         x=st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_symbol][cycle].index, name='Actual'))

        spread = st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_symbol][cycle] - st.session_state.df_20_50_smoothed[st.session_state.df_20_50_smoothed['Sector']==sector_symbol][cycle]
        fig.add_trace(go.Scatter(y=(spread-spread.mean())/spread.std(), x=spread.index, name="Spread Standardized"))

fig.layout.template="plotly"
st.plotly_chart(fig)
if market != "america":
    st.write(info[info.sector==sector][['name','sector','close','volume','market_cap_basic']])

if market == 'america':
    x = st.selectbox(label='Fundamental Ratio on X axis:',
                       options =info.columns[4:],
                       key='x')
    y = st.selectbox(label='Fundamental Ratio on Y axis:',
                       options = info.columns[4:],
                       key='y')

    df, fig = US_fundamentals(country=market, sector=sector,x=x,y=y)
    fig.layout.template="plotly"
    st.plotly_chart(fig)
    st.write(df)
    
