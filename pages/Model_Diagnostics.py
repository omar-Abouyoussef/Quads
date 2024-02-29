import numpy as np
import pandas as pd
import streamlit as st

def goodness_of_fit(X):
    return np.sum(np.sum(X**2, axis=1))/X.shape[0]


loadings=pd.DataFrame(st.session_state.cfa.loadings_,index=st.session_state.performance.columns,columns=["Factor 1","Fctor 2","Factor 3"])
loadings['Communalities'] = st.session_state.cfa.get_communalities()
st.header("Factor Loadings")
st.table(loadings.round(3))
st.write(f"Goodness of fit: {(goodness_of_fit(loadings)*100).round(3)}%")
st.write(f"Goodness of fit: {np.sum(loadings['Communalities']/loadings.shape[0]).round(3)}%")

