# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 20:24:55 2023

@author: USER
"""

import streamlit as st
from function import load_data

from Tabs import home, predict, custom

Tabs = {
        "Home" : home,
        "Prediction" : predict,
        "Custom Model" : custom
        }

#sidebar
st.sidebar.title("Navigation")

#option
page = st.sidebar.radio("Pages", list(Tabs.keys()))

#load data
data,x,y = load_data()

#call function
# if page in ["Prediction", "Visualisation"]:
#     Tabs[page].app(data,x,y)
# else:
#     Tabs[page].app()

Tabs[page].app()