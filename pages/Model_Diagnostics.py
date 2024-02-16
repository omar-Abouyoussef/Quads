import numpy as np
import pandas as pd
import streamlit as st
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecificationParser
from Performance_Quadrants import performance, cfa

def goodness_of_fit(X):
    return np.sum(np.sum(X**2, axis=1))/X.shape[0]


loadings = pd.DataFrame(cfa.loadings_, index=performance.columns, columns=["Short-term","Medium-term", "Long-term"])
st.table(loadings.round(3))
st.write(goodness_of_fit(loadings).round(3))
