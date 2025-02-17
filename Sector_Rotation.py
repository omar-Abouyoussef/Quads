from datetime import datetime as dt
import time
from tvDatafeed import TvDatafeed, Interval
from tradingview_screener import Query, Column
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from retry import retry
import streamlit as st

def get_market_info(market):
    market_info = (Query().select('country','name','exchange','market','sector','close', 'volume').
                   where(Column('volume') > 5000).
                set_markets(market).
                limit(20000).
                get_scanner_data())[1]
    if market == 'egypt':
        infot = market_info
        infor = pd.read_csv('egx_companies.csv')
        info = pd.merge(infot[['name','exchange']], infor, left_on=infot.name, right_on=infor.name).set_index("key_0").reset_index()
        info = info[['name_x','exchange_y','sector']]
        info.columns =  ['name', 'exchange', 'sector']
    else:
        info = market_info
    return info


@retry((Exception), tries=10, delay=1, backoff=0)
@st.cache_data  #add date param
def get_price_data(symbol,exchange,interval,n_bars, date):
    tv = TvDatafeed()
    response = tv.get_hist(symbol=symbol,
                    exchange=exchange,interval=interval, n_bars=n_bars)['close']
    return pd.DataFrame(response)


def get_index_data(market, interval, n_bars):
    info = get_market_info(market)


    sectors = info.sector.unique() 
    # st.session_state.sectors = sectors
    
    symbols = info.name.unique()

    symbol_exchange = pd.DataFrame(list(info.exchange), list(info.name)).reset_index()
    symbol_exchange.columns = ['symbol', 'exchange']
    symbol_exchange.drop_duplicates(subset='symbol',inplace=True)
    symbol_exchange= dict(symbol_exchange.values)


    periods = [20,50,100]

    above_20ma ={}
    above_50ma ={}
    above_100ma ={}

    above_mas = {20:above_20ma, 50:above_50ma, 100:above_100ma}
    # above_mas = {20:above_20ma}

    for period, dic in above_mas.items():
        above = {}
        close_price_data = {}
        counter = 0
        for symbol, exchange in symbol_exchange.items():
            close = pd.DataFrame()
            counter+=1
            print(symbol, counter)

            try:
                data = get_price_data(symbol,exchange,interval,n_bars, date)

                close = pd.DataFrame(data['close'])
                close_price_data[symbol] = data['close']
                close[f'{period}ma'] = close.rolling(period).mean()
                close.dropna(inplace=True)
                close[f'above{period}'] = (close['close'] > close[f'{period}ma']).astype(int)             
                above[symbol] = close[f'above{period}']

            except:
                pass            

        above = pd.DataFrame(above)
        pctabove = above.apply(np.mean,axis = 1)
        dic['Market'] = pctabove * 100
        for sector in sectors:
            sector_symbols = info[info['sector']==sector]['name']
            pctabove = above.loc[:,sector_symbols].apply(np.mean,axis = 1)
            dic[sector] = pctabove * 100
    st.session_state.close_price_data = pd.DataFrame(close_price_data)
    return above_mas


def to_dataframe(dics):
    dfs = []
    for period , dic in dics.items():
        dfs.append(pd.DataFrame(dics[period]).round(2)) 
    df20, df50, df100 = dfs
    return df20, df50, df100

# US INDEX data directly
@retry((Exception), tries=10, delay=1, backoff=0)
@st.cache_data
def get_data_us(sector, suffix,n,freq, date):
    tv = TvDatafeed()
    response = tv.get_hist(symbol=f'{sector}{suffix}',
                    exchange='INDEX',interval=freq,
                    n_bars=n)['close']
    return response

def denoise(x):
    result = seasonal_decompose(x,model="multiplicative", period=30)
    return result.trend



############
#streamlit


st.set_page_config(page_title="Sector Rotation", layout='wide')
st.title('Sector Rotation')

##############################
#inputs
##########################

#inputs

# options = ['america','canada', 'uk', 'germany','uae', 'ksa', 'egypt'],

market = st.selectbox(label='Country:',
                       options = ['america','uae', 'ksa', 'egypt'],
                       key='market')
market = st.session_state.market

historical = st.selectbox(label='Historical:',
                       options = ['No', 'Yes'],
                       key='historical')
historical = st.session_state.historical

plot_type = st.selectbox(label='Plot Type:',
                       options = ['Short-term|Medium-term', 'Medium-term|Long-term'],
                       key='plot_type')
plot_type = st.session_state.plot_type


date = dt.today().date()



info = get_market_info(market=market)
pie = pd.DataFrame(info.value_counts(subset='sector'))


fig2 = go.Figure(
    data = [go.Pie(
        values=pie['count'],
        labels=pie.index,
        hole =0.4, pull = [0.1]*len(pie),
        name = 'Market'
                   )]
    )
fig2. update_layout(showlegend=False)


if market == 'america':

    sectors = ['MM', 'SB', 'SE', 'SF', 'SI', 'SK', 'SL', 'SP', 'SS', 'SU', 'SV', 'SY']
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


        day_20_fastma_response = get_data_us(sector,day_20_suffix,n_days,frequencies[0],date)

        day_50_fastma_response = get_data_us(sector,day_50_suffix,n_days,frequencies[0],date)


        data = pd.concat([day_20_fastma_response, day_50_fastma_response], axis = 1)
        data['Sector'] = [sector] * n_days
        
        df_20_50 = pd.concat([df_20_50, data],axis = 0)
        st.write(df_20_50)
        df_20_50.iloc[:,:-1] = df_20_50.iloc[:,:-1].apply(denoise, axis=0)

    df_20_50.columns = ['Short-term', 'Medium-term', 'Sector']
    df_20_50.index = pd.to_datetime(df_20_50.index).date



    n_months=100

    df_50_100 = pd.DataFrame()
    for idx, sector in enumerate(sectors):    


        day_50_ma_response = get_data_us(sector,day_50_suffix,n_months,frequencies[2],date)

        day_100_ma_response = get_data_us(sector,day_100_suffix,n_months,frequencies[2],date)


        data = pd.concat([day_50_ma_response,day_100_ma_response], axis = 1)
        data['Sector'] = [sector] * n_months
        
        df_50_100 = pd.concat([df_50_100, data],axis = 0)

    df_50_100.columns = ['Medium-term', 'Long-term', 'Sector']
    df_50_100.index = pd.to_datetime(df_50_100.index).date


    ###########
    ####################

    sectors = pd.read_excel('sectors.xlsx', sheet_name = 'Sheet1')

else:
    interval = Interval.in_daily
    n_bars = 1500


    dics = get_index_data(market, interval, n_bars)


    
    df20, df50, df100= to_dataframe(dics)
    


    shortterm = df20.melt(ignore_index=False, var_name='Sector', value_name='Short-term')
    mediumterm = df50.melt(ignore_index=False, var_name='Sector', value_name='Medium-term')
    longterm = df100.melt(ignore_index=False, var_name='Sector', value_name='Long-term')
    


    shortterm['date'] = shortterm.index

    mediumterm['date'] = mediumterm.index

    longterm['date'] = longterm.index

    df_20_50 = pd.merge(left=shortterm, right=mediumterm, how='inner')
    df_20_50.set_index(mediumterm.index, inplace=True)
    df_20_50 = df_20_50[['Short-term','Medium-term', 'Sector']]


    df_50_100 = pd.merge(left=mediumterm, right=longterm, how='inner')
    df_50_100.set_index(longterm.index, inplace=True)
    df_50_100 = df_50_100[['Medium-term','Long-term', 'Sector']]


st.session_state.df_20_50 = df_20_50
st.session_state.df_50_100 = df_50_100




plot = [df_20_50, df_50_100]
if historical == 'No':
    if plot_type == 'Short-term|Medium-term':
        group = plot[0].groupby('Sector').tail(1)
        fig = px.scatter(data_frame=group, x='Medium-term', y='Short-term',
                         title=plot_type, color=sectors.name if market == 'america' else group.Sector,
                         template='plotly_white')


    elif plot_type == 'Medium-term|Long-term':
        group = plot[1].groupby('Sector').tail(1)
        fig = px.scatter(data_frame=group, x='Long-term', y='Medium-term',
                         title=plot_type,color=sectors.name if market == 'america' else group.Sector,
                         template='plotly_white')



    fig.update_traces(marker=dict(size=7))
else:

    last_n = st.slider("Last data points", 1, 52, 5)

    if plot_type == 'Short-term|Medium-term':

        fig = go.Figure()
        for sector in sectors.symbol if market=='america' else df_20_50.Sector.unique():
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
                                        name=sectors[sectors.symbol==sector]['name'].values[0] if market=='america' else df_20_50[df_20_50.Sector==sector]['Sector'].values[0]
                                    )
                                    
                                )

        fig.update_layout(template='plotly_white', width=1000, height=800)
        fig.update_legends()
        fig.add_hline(y=50)
        fig.add_vline(x=50)

    elif plot_type == 'Medium-term|Long-term':

        fig = go.Figure()
        for sector in sectors.symbol if market == 'america' else df_50_100.Sector.unique():
            data = df_50_100[df_50_100['Sector']==sector]
            if market != 'america':
                data = data.resample('M').last()

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
                                        name=sectors[sectors.symbol==sector]['name'].values[0] if market=='america' else df_20_50[df_20_50.Sector==sector]['Sector'].values[0]
                                    )
                                    
                                )

fig.update_layout(template='plotly_white', width=1000, height=800)
fig.update_legends()
fig.add_hline(y=50)
fig.add_vline(x=50)  


container = st.container()
with container:
    plot1, plot2 = st.columns([0.6, 0.4])
    
    with plot1:
        st.plotly_chart(fig)
    with plot2:
        st.plotly_chart(fig2)

st.markdown("""This graph offers valuable insight into sector rotation on a daily and monthly basis. This models the full cycle of market sectors and its sentiment. Sectors that are in the Bottom-Left Quadrant rotate in a clockwise manner till returning back again, this Quadrant also offers the best opportunities to capture a big a move. This is especially true when setting the cycle option to "Medium-term|Long-term" and Historical option to "Yes"  
 \n\n  ***Top-right Quadrant:*** Siginfies extremely bullish and violent movement in price- suited for momentum plays. \n\n  ***Bottom-right:*** After a bullish move sectors weakened and price started to drop.  \n\n ***Bottom-Left Quadrant:*** Falling sector. \n\n  ***Top-left:*** Falling sector started to improve their preformance attracting more buyers.
 \n\n Check Index Tracking page.""")
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=sectors.name if market == 'america' else df_20_50.Sector.unique(),
    y=df_20_50.groupby(by='Sector').tail(1)['Short-term'],
    name='Short-term',
    marker_color='indianred',
    orientation='v'
))
fig3.add_trace(go.Bar(
    x=sectors.name if market == 'america' else df_20_50.Sector.unique(),
    y=df_20_50.groupby(by='Sector').tail(1)['Medium-term'],
    name='Medium-term',
    marker_color='lightsalmon',
    orientation='v'
))

fig3.add_trace(go.Bar(
    x=sectors.name if market == 'america' else df_50_100.Sector.unique(),
    y=df_50_100.groupby(by='Sector').tail(1)['Long-term'],
    name='Long-term',
    marker_color='rgb(55, 83, 109)',
    orientation='v'
))

fig3.add_hline(y=20, line_width=1, line_dash="dash")
fig3.add_hline(y=50, line_width=1, line_dash="dash")
fig3.add_hline(y=80, line_width=1, line_dash="dash")
fig3.update_layout(barmode='group', xaxis_tickangle=-45, width=1000, height=800)

st.plotly_chart(fig3)
st.markdown(""" Bar chart display of all the cycles. """)
#################
##############################
#Heatmap
#############################
##################################





# matrix_type = st.selectbox(label='Heatmap:',
#                        options = ['Correlation', 'Beta'],
#                        key='matrix_type')
# matrix_type = st.session_state.matrix_type


# cycle = st.selectbox(label='Cycle:',
#                        options = ['Short-term','Medium-term', 'Long-term'],
#                        key='cycle')
# cycle = st.session_state.cycle




# if cycle == 'Long-term':
#     pivot = np.log(df_50_100.pivot(columns='Sector')[cycle]).diff().dropna()
# else:
#     pivot = np.log(df_20_50.pivot(columns='Sector')[cycle]).diff().dropna()

# pivot.columns = sectors.name


# if matrix_type == 'Correlation':
#     cor_matrix = pivot.corr()

#     mask = np.zeros_like(cor_matrix, dtype=bool)
#     mask[np.triu_indices_from(mask)] = True
#     # Viz
#     cor_matrix_viz = cor_matrix.mask(mask).dropna(how='all')
#     fig = px.imshow(cor_matrix_viz.iloc[:,:-1].round(2),labels=dict(x="", y="", color="Correlation"),
#                     color_continuous_scale='RdBu_r', text_auto=True, aspect = 'auto')
# else:
#     cov_matrix = pivot.cov()
#     beta_matrix = cov_matrix / pivot.var()
    
#     mask = np.zeros_like(cov_matrix, dtype=bool)
#     mask[np.triu_indices_from(mask)] = True

#     beta_matrix_viz = beta_matrix.mask(mask).dropna(how='all').round(2)
#     fig = px.imshow(beta_matrix_viz.iloc[:,:-1].round(2),labels=dict(x="", y="", color="Beta"),
#                     color_continuous_scale='RdBu_r', text_auto=True, aspect = 'auto')
# st.plotly_chart(fig)
