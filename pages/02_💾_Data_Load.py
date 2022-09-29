# native imports
import datetime

# common imports
import pandas as pd

# special imports
from helper_functions import *

# front-end imports
import streamlit as st

# ===========================================
# Setup
# ===========================================

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

st.title('Data Loading')
st.subheader("The first step towards data analysis is import some data.")
st.markdown("#### There are two options: \n"
            "> ##### 1. Demo data from Kaggle \n"
            "> ##### 2. Your **.csv** file or")

st.markdown("------------------------------------------")
st.markdown("#### Option 1)")
demo = st.checkbox("Click here for uploading and using a demo data",)

intro_text = """
    This is a real industrial dataset uploaded at Kaggle.
    Data comes from a real mining flotation plant in Brazil.
    
    You can access the dataset <a href = "https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process"> 
    here </a>, and get more details about it.
    """
intro = st.expander("Click here for more info about the dataset")
if demo:
    link = "https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process?select=MiningProcess_Flotation_Plant_Database.csv"
    data_load_state = st.text('Loading data...')

    dfdemo = getDataFromKaggle(link)
    dfdemo.copy()
    data_load_state.text("Great! Data loaded successfully!")
    
    # Caching data
    if "dfdemo" not in st.session_state:
        st.session_state.dfdemo = dfdemo

    
    # Display data frame
    st.dataframe(dfdemo)

    st.markdown(
        "To continue with your journey, go to the following page: **Data Preparation**.")


with intro:
    sub_text(intro_text)


st.markdown("------------------------------------------")
st.markdown("#### Option 2)")
st.markdown("##### First of all, please pay attention to the format of your *.csv file")
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

st.markdown(
    "##### Second, before selecting the file that will be uploaded, select the formats used to generate your CSV Sfile.")

# User specification for the csv file format
separator_format = st.selectbox(
    "Choose the Separator format used in your CSV file", (";", ","))
decimal_format = st.selectbox(
    "Choose the Decimal separator format used in your CSV file", (".", ","))

st.markdown(
    "##### Now, select your file that will be uploaded.")

uploaded_file = st.file_uploader("",
    type="csv",
    key='uploaded_file')

if uploaded_file:

    data_load_state = st.text('Loading data...')

    df = getDataFromCSV(uploaded_file, separator_format, decimal_format)
    df.copy()
    data_load_state.text("Great! Data loaded successfully!")
    
    # Caching data
    if "df" not in st.session_state:
        st.session_state.df = df

    
    # Display data frame
    st.dataframe(df)

    st.markdown(
        "To continue with your journey, go to the following page: **Data Preparation**.")

