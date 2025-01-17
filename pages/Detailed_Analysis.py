import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import statsmodels.api as sm
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split, FixedThresholdClassifier
from sklearn.metrics import confusion_matrix, classification_report



"""### Detailed Analysis \n\n\n\n"""
X=st.session_state.X
y= st.session_state.y
X_test = X.tail(250)
y_test = y.tail(250)

st.write(f'{y.name}')
features = st.session_state.features

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.4, shuffle=True, random_state=1)

sm.add_constant(X)
regression = sm.OLS(endog=y_train,exog=X_train).fit()
st.write(regression.summary().tables[0].as_html(), unsafe_allow_html=True)
st.write(regression.summary().tables[1].as_html(), unsafe_allow_html=True)
st.write(regression.summary().tables[2].as_html(), unsafe_allow_html=True)


regression_tree = DecisionTreeRegressor(max_depth=10, max_leaf_nodes=30, min_samples_split=50, splitter='best')       #max_depth=10, max_leaf_nodes=30, min_samples_split=50, splitter='best')
regression_tree.fit(X_train,y_train)
y_hat = regression_tree.predict(X_test)
fig = px.scatter(x=y_hat,y=y_test, hover_data=[y_test.index.date], trendline='ols', template='seaborn')
fig.update_layout(xaxis_title='Prediction', yaxis_title='Actual Returns')
fig.update_yaxes(tickformat=".2%")
fig.update_xaxes(tickformat=".2%")
fig.add_hline(y=0)
st.plotly_chart(fig,theme=None)
y_pred = regression_tree.predict(st.session_state.current.values.reshape((1,-1)))
st.write(f"Expected Return: {np.round(y_pred,4)*100}%")


threshold = st.slider(label='Certainty',
                      min_value=0.01,
                      max_value=0.99,
                      step=0.01,
                      value=0.5)


if threshold:
    y_binned = (y>0).astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X,y_binned, test_size=0.4, shuffle=True, random_state=1, stratify=y_binned)
    base_classifier = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10, min_samples_leaf=10, splitter='best').fit(X_train,y_train)     #max_depth=10, max_leaf_nodes=20    better threshold
    
    clf = FixedThresholdClassifier(estimator=base_classifier,threshold=threshold)
    clf.fit(X_train,y_train)                                                            #max_depth=20, max_leaf_nodes=30
    st.write(f'Train: {clf.score(X_train,y_train):.2f}')
    st.write(f'Test: {clf.score(X_test,y_test):.2f}')
    yhat = clf.predict(X_test)
    report_df = pd.DataFrame(classification_report(y_test, yhat,output_dict=True))
    report_df.columns = ['Bearish','Bullish','accuracy','macro avg','weighted avg']
    st.write(report_df.T)
    y_prob = clf.predict_proba(st.session_state.current.values.reshape((1,-1)))
    st.write(f'Probability of Bullish Move:{y_prob[:,1].round(3)*100}%')
