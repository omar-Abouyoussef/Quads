import numpy as np
import pandas as pd
import streamlit as st

def goodness_of_fit(X):
    return np.sum(X)/X.shape[0]


st.session_state.loadings['Communalities'] = st.session_state.cfa.get_communalities()
st.header("Factor Loadings")
st.table(st.session_state.loadings.round(3))
st.write(f"Goodness of fit: {(goodness_of_fit(loadings['Communalities'])*100).round(3)}%")
