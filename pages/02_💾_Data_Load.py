# native imports
import datetime

# common imports
import pandas as pd

# special imports
from utils import *

# front-end imports
import streamlit as st


st.set_page_config(
    # Can be "centered" or "wide". In the future also "dashboard", etc.
    layout="centered",
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    # String or None. Strings get appended with "• Streamlit".
    page_title="Sientia - SPPE Module",
    page_icon='💾',  # String, anything supported by st.image, or None.
)


##########################
### App page beginning ###
##########################
# Início da página Data Preparation
st.title('Loading and preparing data!')
st.subheader(
    "The first step towards data analysis is import the **.csv** file that contains the data.")

st.info(
    """
    **Attention** to the **standard formatting** of the **.csv file**:
        \n* Delimiter = "," or ";"
        \n* Decimal = "." or ","
        \n* Encoding = "UTF-8"
        \n* Datetime = "%Y-%m-%d %H:%M:%S"
        \n* Variable "Time" or "Time Stamp" as column 0 of the dataframe         
    """
)

st.subheader(
    "Before selecting the file that will be uploaded, select the format used to generate your CSv file.")

# User specification for the csv file format
separator_format = st.selectbox("Choose the Separator format used in your CSV file",(";", ","))
decimal_format = st.selectbox("Choose the Decimal separator format used in your CSV file",(".", ","))

uploaded_file = st.file_uploader(
    "Upload your csv file here",
    type="csv",
    key='uploaded_file')

if uploaded_file:

    data_load_state = st.text('Loading data...')

    df = getDataFromCSV(uploaded_file, separator_format, decimal_format).copy()

    data_load_state.text("Great! Data loaded successfully!")
    st.markdown(
        "Para continuar, navegue pela aba **lateral esquerda** e selecione as opções.")
