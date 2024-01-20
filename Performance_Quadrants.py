import datetime as dt
import pandas as pd
import requests
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
import investpy
import streamlit as st

@st.cache_data
def change(data, freq):
    return data.iloc[-1]/data.iloc[-(freq+1)] - 1

   
@st.cache_data   
def get_data(market:str, stock_list:list, start:dt.date, end:dt.date, key:str):
    

    close_prices = pd.DataFrame(columns=stock_list)
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


######################
####################

st.set_page_config(page_title="Performance Quadrant", layout="wide")
st.title('Performance Quadrant')
st.sidebar.header('Home')



#####
today = dt.date.today()
start = today - dt.timedelta(365)

#inputs
country = st.selectbox(label='Country:',
                       options = ['Egypt', 'United States', 'Saudi Arabia'],
                       key='country')
country = st.session_state.country

plot = st.selectbox(label='Plot type:',
                    options=['Short-term|Medium-term', 'Short-term|Long-term', 'Medium-term|Long-term'],
                    key='plot')
plot = st.session_state.plot


stock_list = investpy.stocks.get_stocks_list(country = country)
codes = {'Egypt':'EGX', 'United States':'US', 'Saudi Arabia':'SR'}

close_prices = get_data(market = codes[country], stock_list=stock_list,
                        start=start, end=today, key=st.secrets["eod_api_key"])
    

close_prices.dropna(axis = 1, inplace = True)    

one_day_return = change(close_prices, 1)
two_day_return = change(close_prices, 2)
three_day_return = change(close_prices, 3)
weekly_return = change(close_prices, 5)
three_week_return = change(close_prices, 15)
one_month_return = change(close_prices, 22)
three_month_return = change(close_prices, 66)
six_month_return = change(close_prices, 132)


performance = pd.DataFrame(
list(
    zip(one_day_return,
        two_day_return,
        three_day_return,
        weekly_return,
        three_week_return,
        one_month_return,
        three_month_return,
        six_month_return)
        ),
        columns = ['1-Day', '2-Day', '3-Day', '1-Week', '3-Week', '1-Month', '3-Month', '6-Month'],
        index = close_prices.columns
        )


scaler = StandardScaler()
performance_scaled = scaler.fit_transform(performance)
fa = FactorAnalysis(n_components = 3, rotation = "varimax", random_state=0)
factors = fa.fit_transform(performance_scaled)
factors = pd.DataFrame(data = factors, index = performance.index, columns = ['Short-term', 'Medium-term', 'Long-term'])

model = KMeans(n_clusters = 4, random_state=0)
labels = model.fit_predict(factors)

factors['performance']=labels
factors['performance'] = factors['performance'].map({0:'Weakening', 1:'Improving', 2:'Falling', 3:'Momentum'})

if plot == 'Short-term|Medium-term':
    fig = px.scatter(factors, x='Medium-term', y='Short-term',
                     hover_data = [factors.index], color='performance')

elif plot == 'Medium-term|Long-term':
    fig = px.scatter(factors, x='Long-term', y='Medium-term',
                     hover_data = [factors.index], color='performance')
else: 
   fig = px.scatter(factors, x='Long-term', y='Short-term',
                    hover_data = [factors.index], color='performance')

fig.add_hline(y=0)
fig.add_vline(x=0)

cols = st.columns([0.7,0.3])
with cols[0]:
    st.plotly_chart(fig)
with cols[1]:
    st.dataframe(factors,column_order=('','performance'))
    

