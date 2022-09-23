# native imports
import datetime

# common imports
import pandas as pd

# special imports


# front-end imports
import streamlit as st


st.set_page_config(
    # Can be "centered" or "wide". In the future also "dashboard", etc.
    layout="centered",
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    # String or None. Strings get appended with "• Streamlit".
    page_title="Sientia - SPPE Module",
    page_icon='📊',  # String, anything supported by st.image, or None.
)

# ===========================================
# Functions Data Retrieval
# ===========================================
