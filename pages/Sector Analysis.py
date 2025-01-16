import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf
import datetime as dt
from tradingview_screener import Query, Column 
from tvDatafeed import TvDatafeed, Interval
from retry import retry
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split, FixedThresholdClassifier
from sklearn.metrics import confusion_matrix, classification_report
import dtreeviz 
import base64


# ###################
# Functions
##############################
def get_market_info(market):
    market_info = (Query().select('country','name','exchange','market','sector','close','volume','market_cap_basic').
                   where(Column('volume') > 5000).
                set_markets(market).
                limit(20000).
                get_scanner_data())[1]
    if market == 'egypt':
        infot = market_info
        infor = pd.read_csv('egx_companies.csv')
        #info = pd.concat(infot[['name','exchange','close','volume','market_cap_basic']], infor.sector], axis=1, join='inner')
        info = pd.merge(left=infot[['name','exchange','close','volume','market_cap_basic']], right=infor, on='name')
        info = info[['name','sector','close','volume','market_cap_basic']]
    else:
        info = market_info
    return info


def scale(x):
    x_scaled = (x - x.mean())/x.std()
    return x_scaled
################################
########################################3

@retry((Exception), tries=10, delay=1, backoff=0)
@st.cache_data(ttl=dt.timedelta(hours=4.0))
def get_index_data(sector, suffix,n,freq,date):
    tv = TvDatafeed()
    response = tv.get_hist(symbol=f'{sector}{suffix}',
                    exchange='INDEX',interval=freq,
                    n_bars=n)['close']
    return response

@retry((Exception), tries=10, delay=1, backoff=0)
@st.cache_data(ttl=dt.timedelta(hours=4.0))
def get_stock_data(symbol, exchange,n,freq,date):
    tv = TvDatafeed()
    response = tv.get_hist(symbol=symbol,
                    exchange=exchange,interval=freq,
                    n_bars=n)['close']
    return response


def svg_write(svg, center=True):
    """
    Disable center to left-margin align like other objects.
    """
    # Encode as base 64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
    html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'

    # Write the HTML
    st.write(html, unsafe_allow_html=True)


###########################################
##########################################







date = dt.datetime.today().date()

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
    
    if standardize == 'Yes':
        series = st.session_state.df_50_100[st.session_state.df_50_100['Sector']==sector_symbol][cycle]
        zscore = (series - series.rolling(window=60).mean()) / series.rolling(window=60).std()
        
        fig = px.line(zscore, line_shape="spline")
        fig.add_hline(1.27, line_width=1, line_dash="dash")
        fig.add_hline(-1.27, line_width=1, line_dash="dash")


    else:
        series = st.session_state.df_50_100[st.session_state.df_50_100['Sector']==sector_symbol][cycle]
        fig = px.line(series,line_shape="spline")


else:
    if standardize == 'Yes':
        series = st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_symbol][cycle]
        zscore = (series - series.rolling(window=60).mean()) / series.rolling(window=60).std()
        
        fig = px.line(zscore,line_shape="spline")

        fig.add_hline(1.27, line_width=1, line_dash="dash")
        fig.add_hline(-1.27, line_width=1, line_dash="dash")


    else:
        series = st.session_state.df_20_50[st.session_state.df_20_50['Sector']==sector_symbol][cycle]
        fig = px.line(series,line_shape="spline")

st.plotly_chart(fig)
if market != "america":
    st.write(info[info.sector==sector][['name','sector','close','volume','market_cap_basic']])

###############################################
###############################################
###############################################
###############################################

if market =='america':

    sector_names_us_dic = {'Basic Materials': 'SB', 'Telecommunications': 'SL', 'Finance': 'SF', 'Industrials': 'SI', 'Technology': 'SK', 'Consumer Staples': 'SP', 'Real Estate': 'SS', 'Utilities': 'SU', 'Health Care': 'SV', 'Consumer Discretionary': 'SY', 'Energy': 'SE', 'Market':'MM'}
    sector_etf_dic = {'MM':'IWM', 'SB':'XLB', 'SL':'XLC', 'SF':'XLF', 'SI':'XLI', 'SK':'XLK', 'SP':'XLP', 'SS':'XLRE', 'SU':'XLU', 'SV':'XLV', 'SY':'XLY', 'SE':'XLE'}


    day_5_suffix = 'FD'
    day_20_suffix = 'TW'
    day_50_suffix = 'FI'
    day_100_suffix = 'OH'
    day_200_suffix = 'TH'

    in_monthly = Interval.in_monthly
    in_weekly = Interval.in_weekly
    in_daily = Interval.in_daily
    frequencies = [in_daily, in_weekly, in_monthly]
    n=2500
    fcast_n=30
    freq = frequencies[0]

    etf = us_sectors[us_sectors['name']==sector]['etf'].values[0]
    st.session_state.etf = etf 
    
    data = pd.DataFrame(get_stock_data(symbol=etf,exchange='AMEX',n=n,freq=freq,date=date))
    data.columns = ['etf']

    data['sector_long_term_sentiment'] = get_index_data(sector=sector_names_us_dic[sector],suffix=day_100_suffix,n=n,freq=freq,date=date)
    data['sector_short_term_sentiment'] = get_index_data(sector=sector_names_us_dic[sector],suffix=day_20_suffix,n=n,freq=freq,date=date)
    data['market_short_term_sentiment'] = get_index_data(sector='MM',suffix=day_20_suffix,n=n,freq=freq,date=date)
    data['market_long_term_sentiment'] = get_index_data(sector='MM',suffix=day_100_suffix,n=n,freq=freq,date=date)

    data['rinf'] = get_stock_data(symbol='RINF',exchange='AMEX',n=n,freq=freq,date=date)
    data['uvxy'] = get_stock_data(symbol='UVXY',exchange='AMEX',n=n,freq=freq,date=date)
    data['shy'] = get_stock_data(symbol='SHY',exchange='NASDAQ',n=n,freq=freq,date=date)
    # data['gld'] = get_stock_data(symbol='GLD',exchange='NASDAQ',n=n,freq=freq,date=date)

    df_ret = pd.DataFrame()
    df =  data.copy()
    df_ret[['etf', 'bond_price_change','inflation_expectation_change', 'vix_change']] = (df[['etf', 'shy', 'rinf','uvxy']]).pct_change(fcast_n)
    # df_ret[['sector_long_term_sentiment_change','sector_short_term_sentiment_change',
    #        'market_short_term_sentiment_change','market_long_term_sentiment_change','inflation_expectation_change']] = (df[['sector_long_term_sentiment','sector_short_term_sentiment',
                                                                                            #        'market_short_term_sentiment','market_long_term_sentiment', 'rinf']]+100).pct_change(30)
    df_ret[['sector_short_term_sentiment_change',
        'market_short_term_sentiment_change']] = (df[['sector_short_term_sentiment',
                                                      'market_short_term_sentiment']]+100).pct_change(fcast_n)


    df_ret[['sector_long_term_sentiment','sector_short_term_sentiment',
            'market_short_term_sentiment','market_long_term_sentiment']] = data[['sector_long_term_sentiment','sector_short_term_sentiment',
                                                                                                'market_short_term_sentiment','market_long_term_sentiment']]
    df_ret['etf'] = df_ret['etf'].shift(-fcast_n)

    df_final = df_ret.iloc[fcast_n:-fcast_n,].dropna()


    X=df_final.drop('etf',axis=1)
    y = df_final['etf']
    y.name = etf
    features = X.columns
    
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.features = features


    switch_page = st.button("Detailed Analysis")
    page_file = 'pages/Detailed_analysis.py'

    if switch_page:
        # Switch to the selected page

        st.switch_page(page_file)
    
    else:
        y_binned = (y>0).astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(X,y_binned, test_size=0.4, shuffle=True, random_state=1, stratify=y_binned)

        base_classifier = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10, min_samples_leaf=10, splitter='best')      #max_depth=10, max_leaf_nodes=20    better threshold
        base_classifier.fit(X_train,y_train)





        color_blind_friendly_colors = [
        None,  # 0 classes
        None,  # 1 class
        ['#FF4B4B', '#04599A'],  # 2 classes
        ]
        COLORS = {'highlight': 'ORANGE', 'classes': color_blind_friendly_colors}

        viz_model = dtreeviz.model(model = base_classifier, X_train=X_train, y_train=y_train, feature_names=X_train.columns, class_names=['BEARISH','BULLISH'],target_name='Expectation')
        current = df_ret.dropna(subset=features).iloc[-1,1:]
        tree_fig = viz_model.view(x=current, show_just_path=False, show_root_edge_labels=False, show_node_labels=False,leaftype='pie',max_X_features_TD=2, colors=COLORS).svg()

        f"""##### Expected {fcast_n}-Day Move"""
        svg_write(tree_fig)        

