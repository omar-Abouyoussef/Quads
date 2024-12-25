from datetime import datetime as dt
from tvDatafeed import TvDatafeed, Interval
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


st.set_page_config(page_title="Sector Rotation")
st.title('Sector Rotation')

##############################
#inputs
##########################

#inputs
historical = st.selectbox(label='Historical:',
                       options = ['No', 'Yes'],
                       key='historical')
historical = st.session_state.historical

plot_type = st.selectbox(label='Plot Type:',
                       options = ['Short-term|Medium-term', 'Medium-term|Long-term' ],
                       key='plot_type')
plot_type = st.session_state.plot_type






sectors = ['S5', 'SK', 'SE', 'SS', 'SF', 'SI', 'SB', 'SP', 'SY', 'SU', 'SL', 'SV']
day_5_suffix = 'FD'
day_20_suffix = 'TW'
day_50_suffix = 'FI'
day_100_suffix = 'OH'
day_200_suffix = 'TH'

in_monthly = Interval.in_monthly
in_weekly = Interval.in_weekly
in_daily = Interval.in_daily
frequencies = [in_daily, in_weekly, in_monthly]


n_days=1000

df_20_50 = pd.DataFrame()
for idx, sector in enumerate(sectors):    


    day_20_fastma_response = get_data(sector,day_20_suffix,n_days,frequencies[0],date)

    day_50_fastma_response = get_data(sector,day_50_suffix,n_days,frequencies[0],date)


    data = pd.concat([day_20_fastma_response, day_50_fastma_response], axis = 1)
    data['Sector'] = [sector] * n_days
    
    df_20_50 = pd.concat([df_20_50, data],axis = 0)

df_20_50.columns = ['Short-term', 'Medium-term', 'Sector']
df_20_50.index = pd.to_datetime(df_20_50.index).date



n_months=100

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


    fig.add_vline(x=50)
    fig.add_hline(y=50)
    st.plotly_chart(fig)

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

        fig.update_layout(template='plotly_white')
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

        fig.update_layout(template='plotly_white')
        fig.update_legends()
        fig.add_hline(y=50)
        fig.add_vline(x=50)
     
    st.plotly_chart(fig)    
