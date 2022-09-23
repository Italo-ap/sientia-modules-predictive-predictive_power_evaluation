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
    page_icon='💾',  # String, anything supported by st.image, or None.
)

# ===========================================
# Functions Data Retrieval
# ===========================================


@st.cache(allow_output_mutation=True)
def getDataFromCSV(file) -> pd.DataFrame:
    dataFrame = pd.read_csv(file, sep=",", decimal=".",
                            encoding="UTF-8",
                            index_col=0,
                            low_memory=False)

    dataFrame.index = pd.to_datetime(
        dataFrame.index, format='%Y-%m-%d %H:%M:%S')
    dataFrame = dataFrame.sort_index(ascending=True)
    dataFrame = dataFrame.apply(pd.to_numeric, errors='coerce')
    return dataFrame

##########################
### App page beginning ###
##########################
# Início da página Data Preparation
st.title('Loading and preparing data!')
st.markdown(
    "The first step is to import the **.csv** file that contains the data you want to analyze.")

st.info(
    """
    Attention to the **standard formatting** of the **.csv file**:
        \n* Delimiter = ","
        \n* Decimal = "."
        \n* Encoding = "UTF-8"
        \n* Datetime = "%Y-%m-%d %H:%M:%S"
        \n* Variable "Time" or "Time Stamp" as column 0 of the dataframe         
    """
)

uploaded_file = st.file_uploader(
    "Upload your csv file here",
    type="csv",
    key='uploaded_file')

if uploaded_file:

    data_load_state = st.text('Loading data...')

    df = getDataFromCSV(uploaded_file).copy()

    data_load_state.text("Great! Data loaded successfully!")
    st.markdown(
        "Para continuar, navegue pela aba **lateral esquerda** e selecione as opções.")
