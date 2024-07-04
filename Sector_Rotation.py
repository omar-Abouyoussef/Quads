from datetime import datetime as dt
from tvDatafeed import TvDatafeed, Interval
from tradingview_screener import Query, Column, get_all_symbols
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from retry import retry
import streamlit as st


date = dt.today().date()

@retry((Exception), tries=10, delay=1, backoff=0)
@st.cache_data
def get_data(sector, suffix,n,freq, date):
    tv = TvDatafeed()
    response = tv.get_hist(symbol=f'{sector}{suffix}',
                    exchange='INDEX',interval=freq,
                    n_bars=n)['close']
    return response





############
#streamlit


st.set_page_config(page_title="Sector Rotation", page_layout='wide')
st.title('Sector Rotation')

##############################
#inputs
##########################

#inputs


market = st.selectbox(label='Coutry:',
                       options = ['america'],
                       key='market')
market = st.session_state.market

historical = st.selectbox(label='Historical:',
                       options = ['No', 'Yes'],
                       key='historical')
historical = st.session_state.historical

plot_type = st.selectbox(label='Plot Type:',
                       options = ['Short-term|Medium-term', 'Medium-term|Long-term' ],
                       key='plot_type')
plot_type = st.session_state.plot_type






sectors = ['MM', 'SK', 'SE', 'SS', 'SF', 'SI', 'SB', 'SP', 'SY', 'SU', 'SL', 'SV']
day_5_suffix = 'FD'
day_20_suffix = 'TW'
day_50_suffix = 'FI'
day_100_suffix = 'OH'
day_200_suffix = 'TH'

in_monthly = Interval.in_monthly
in_weekly = Interval.in_weekly
in_daily = Interval.in_daily
frequencies = [in_daily, in_weekly, in_monthly]


n_days=52

df_20_50 = pd.DataFrame()
for idx, sector in enumerate(sectors):    


    day_20_fastma_response = get_data(sector,day_20_suffix,n_days,frequencies[0],date)

    day_50_fastma_response = get_data(sector,day_50_suffix,n_days,frequencies[0],date)


    data = pd.concat([day_20_fastma_response, day_50_fastma_response], axis = 1)
    data['Sector'] = [sector] * n_days
    
    df_20_50 = pd.concat([df_20_50, data],axis = 0)

df_20_50.columns = ['Short-term', 'Medium-term', 'Sector']
df_20_50.index = pd.to_datetime(df_20_50.index).date



n_months=24

df_50_100 = pd.DataFrame()
for idx, sector in enumerate(sectors):    


    day_50_ma_response = get_data(sector,day_50_suffix,n_months,frequencies[2],date)

    day_100_ma_response = get_data(sector,day_100_suffix,n_months,frequencies[2],date)


    data = pd.concat([day_50_ma_response,day_100_ma_response], axis = 1)
    data['Sector'] = [sector] * n_months
    
    df_50_100 = pd.concat([df_50_100, data],axis = 0)

df_50_100.columns = ['Medium-term', 'Long-term', 'Sector']
df_50_100.index = pd.to_datetime(df_50_100.index).date


###########
####################3
sector_names = pd.read_excel('sectors.xlsx', sheet_name = 'Sheet1')






plot = [df_20_50, df_50_100]

if historical == 'No':
    if plot_type == 'Short-term|Medium-term':
        group = plot[0].groupby('Sector').tail(1)
        fig = px.scatter(data_frame=group, x='Medium-term', y='Short-term',title=plot_type, color=sector_names.name, template='plotly_white')


    elif plot_type == 'Medium-term|Long-term':
        group = plot[1].groupby('Sector').tail(1)
        fig = px.scatter(data_frame=group, x='Long-term', y='Medium-term',title=plot_type,color=sector_names.name, template='plotly_white')




else:
    last_n = st.slider("Last data points", 1, data.shape[0], 5)

    if plot_type == 'Short-term|Medium-term':

        fig = go.Figure()
        for sector in sectors:
            data = df_20_50[df_20_50['Sector']==sector]
            fig.add_trace(
                                    go.Scatter(
                                        x=data["Medium-term"].tail(last_n),
                                        y=data["Short-term"].tail(last_n),
                                        mode="lines+markers",
                                        line=dict(
                                            shape="spline"
                                        ),
                                        marker=dict(
                                            symbol="arrow",
                                            size=10,
                                            angleref="previous"
                                        ),hovertext=df_20_50.tail(last_n).index,
                                        name=sector_names[sector_names.symbol==sector]['name'].values[0]
                                    )
                                    
                                )

        fig.update_layout(template='plotly_white', width=1000, height=800)
        fig.update_legends()
        fig.add_hline(y=50)
        fig.add_vline(x=50)

    elif plot_type == 'Medium-term|Long-term':

        fig = go.Figure()
        for sector in sectors:
            data = df_50_100[df_50_100['Sector']==sector]
            fig.add_trace(
                                    go.Scatter(
                                        x=data["Long-term"].tail(last_n),
                                        y=data["Medium-term"].tail(last_n),
                                        mode="lines+markers",
                                        line=dict(
                                            shape="spline"
                                        ),
                                        marker=dict(
                                            symbol="arrow",
                                            size=10,
                                            angleref="previous"
                                        ),hovertext=df_50_100.tail(last_n).index,
                                        name=sector_names[sector_names.symbol==sector]['name'].values[0]
                                    )
                                    
                                )

fig.update_layout(template='plotly_white', width=1000, height=800)
fig.update_legends()
fig.add_hline(y=50)
fig.add_vline(x=50)  


fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=df_20_50.groupby(by='Sector').tail(1)['Short-term'],
    y=sector_names.name,
    name='Short-term',
    marker_color='indianred',
    orientation='h'
))
fig2.add_trace(go.Bar(
    x=df_20_50.groupby(by='Sector').tail(1)['Medium-term'],
    y=sector_names.name,
    name='Medium-term',
    marker_color='lightsalmon',
    orientation='h'
))

fig2.add_trace(go.Bar(
    x=df_50_100.groupby(by='Sector').tail(1)['Long-term'],
    y=sector_names.name,
    name='Long-term',
    marker_color='rgb(55, 83, 109)',
    orientation='h'
))

fig2.add_vline(x=20, line_width=1, line_dash="dash")
fig2.add_vline(x=50, line_width=1, line_dash="dash")
fig2.add_vline(x=80, line_width=1, line_dash="dash")
fig2.update_layout(barmode='group', xaxis_tickangle=-45, width=1000, height=800)

container = st.container()
with container:
    plot1, plot2 = st.columns([0.6, 0.4])
    
    with plot1:
        st.plotly_chart(fig)
    with plot2:
        st.plotly_chart(fig2) 

  

tickers = get_all_symbols(market)

df = (Query()
 .select('name', 'sector','close', 'volume', 'SMA20', 'SMA50', 'SMA100')
 .where(
     Column('close') > 20,
     Column('close') < Column('SMA20'),
     Column('close') < Column('SMA50') 
 ).limit(len(tickers))
 .get_scanner_data())[1]

sector_names = df.sector.unique()
sel_sector = st.selectbox(label='Sector:',
                       options = sector_names,
                       key='sel_sector')
sel_sector = st.session_state.sel_sector


st.write('Stocks below their 20 & 50 day average:')
st.write(df[df.sector==sel_sector])
