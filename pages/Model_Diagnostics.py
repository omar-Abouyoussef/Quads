import numpy as np
import pandas as pd
import streamlit as st

def goodness_of_fit(X):
    return np.sum(X)/X.shape[0]


loadings=pd.DataFrame(st.session_state.cfa.loadings_,index=st.session_state.performance.columns,columns=["Factor 1","Factor 2","Factor 3"])
loadings['Communalities'] = st.session_state.cfa.get_communalities()
st.header("Factor Loadings")
st.table(loadings.round(3))
st.write(f"Goodness of fit: {(goodness_of_fit(loadings['Communalities'])*100).round(3)}%")
st.write("Factors represent short, medium, and long-term performance of equties. Loadings matrix represents which latent factors load on the seleceted set of variables.")
st.write("If Factor 1 loads heavily on variables(1-Day, 2-Day, 3-Day) then it might represent the short-term behaviour of the equties.")
st.write("Example: if (Factor 2|1-day) is +0.723 then factor 2 loads positively on 1-day varaible. If (Factor 1|1-Week) is -0.834 then factor 1 loads negatively on that variable and stocks with high-score for factor 1 means the are experiencing a down-trend in medium term")
