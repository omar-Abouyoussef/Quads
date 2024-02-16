import numpy as np
import pandas as pd
import streamlit as st
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecificationParser
from Performance_Quadrants import performance, loadings
model_dict = {"Short-term": ["1-Day", "2-Day", "3-Day"],
"Medium-term": ["1-Week", "2-Week", "3-Week"],
"Long-term": ["1-Month", "3-Month", "6-Month"]}

model_spec = ModelSpecificationParser.parse_model_specification_from_dict(performance, model_dict)
cfa = ConfirmatoryFactorAnalyzer(model_spec)
cfa.fit(performance.values)
loadings = pd.DataFrame(cfa.loadings_, index=performance.columns, columns=["Short-term","Medium-term", "Long-term"])
print(loadings.round(3))
print(goodness_of_fit(loadings).round(3))
