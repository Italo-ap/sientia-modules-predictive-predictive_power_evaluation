# native imports
import datetime

# common imports
import pandas as pd
from rich import print as rprint

# special imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.seasonal import seasonal_decompose

# front-end imports
import streamlit as st

#custom imports
from utils import validate_datetime_index_lastvalue

st.set_page_config(
    # Can be "centered" or "wide". In the future also "dashboard", etc.
    layout="centered",
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    # String or None. Strings get appended with "• Streamlit".
    page_title="Sientia - SPPE Module",
    page_icon='📊',  # String, anything supported by st.image, or None.
)

# ===========================================
# Functions of Data caching
# ===========================================
dfraw = pd.DataFrame.from_dict(st.session_state.df)

# Initializing dfCleaned DataFrame
dfCleaned = pd.DataFrame()
if "dftidy" not in st.session_state:
    st.session_state.dftidy = dfCleaned

##########################
### App page beginning ###
##########################
st.title('Data Selection and Preparation')
st.subheader("Now it is time to select and prepare your data, like filtering, cleaning and transforming.")

st.markdown("#### First of all go to the navigation page on your left and follow through Steps 1 and 2.")

if not dfraw.empty:
    st.sidebar.title("Data Selection")

    expanderFltTags = st.sidebar.expander(
        label='Step 1: Select the varibles for the analysis',
        expanded=False)

    dfTags = dfraw.columns.values.tolist()

    fltTags = expanderFltTags.multiselect(
        label='Selected Variables:',
        options=dfTags,
        # default=fltTags,
        key='fltTagsPreparation')

    if len(fltTags) > 10:
        expanderFltTags.warning(
            'More than **10 TAGs** where selected')

    dfRaw = dfraw[fltTags]

    expanderFltDate = st.sidebar.expander(
        label='Step 2: Select the data range',
        expanded=False)

    startTimeDf = dfraw.index[0]
    
    # Checking if last index value of csv file is correct]
    # Funcition and functionality should be further investigated and deploy to avoid future bugs regarding 'NaT' value in the last row of the CSV file
    last_index_value = validate_datetime_index_lastvalue(dfraw.index[-1])
    
    endTimeDf = dfraw.index[-1]



    fltDateStart = expanderFltDate.date_input(
        label='Initial date:',
        value=startTimeDf)
    fltTimeStart = expanderFltDate.time_input(
        label='Initial hour',
        value=datetime.time(0, 0, 0))

    fltDateEnd = expanderFltDate.date_input(
        label='End date:',
        max_value=fltDateStart + datetime.timedelta(weeks=1),
        value=(fltDateStart + datetime.timedelta(weeks=1)))
    fltTimeEnd = expanderFltDate.time_input(
        label='End hour',
        value=datetime.time(endTimeDf.hour, endTimeDf.minute, endTimeDf.second))

    selStart = datetime.datetime.combine(
        fltDateStart, fltTimeStart)
    selEnd = datetime.datetime.combine(
        fltDateEnd, fltTimeEnd)

    expanderFltDate.warning(
        'Maximum range is 1 week for best performance')

    dfRawRange = dfRaw.loc[selStart:selEnd]
    dfRawRange = dfRawRange

    if fltTags != []:

        st.markdown("------------------------------------------")
        st.markdown(
            "Perfect, now just select the options below to start exploring the data.")

        #################
        ### Dataframe ###
        #################

        showRawData = st.checkbox(
            label='Show data',
            value=False,
            key='showRawData')

        if (showRawData):
            st.dataframe(data=dfRawRange)

            ############
            ### Info ###
            ############

        showInfo = st.checkbox(
            label='Show Dataframe information (format and missing values)', value=False, key='showInfo')
        if (showInfo):

            dfInfo = pd.DataFrame()
            dfInfo["Types"] = dfRawRange.dtypes
            dfInfo["Missing Values"] = dfRawRange.isnull().sum()
            dfInfo["Missing Values % "] = (
                dfRawRange.isnull().sum()/len(dfRawRange)*100)
            # Converting DF as type str to due Arrow bug, as tried before using dfInfo only. Bug reported by streamlit on September 2022.
            st.table(dfInfo.astype(str))

        ################
        ### Cleaning ###
        ################
       

        execCleaning = st.checkbox(label='Do data cleaning', value=(
            False), key='execCleaning')
        if (execCleaning):

            methodCleaning = ["Interpolation", "Drop NaN"]

            selectCleaning = st.selectbox(label='Please select a cleaning method',
                                            options=methodCleaning,
                                            key='selectCleaning')

            if selectCleaning == "Drop NaN":

                dfCleaned = dfRawRange.dropna(how="any")
              

            elif selectCleaning == "Interpolation":

                methodInterpolation = [
                    "linear", "nearest", "zero", "slinear", "quadratic", "cubic"]

                selectInterpolation = st.selectbox(label='Select a method for interpolation',
                                                    options=methodInterpolation,
                                                    key='selectInterpolation')

                dfCleaned = dfRawRange.interpolate(
                    method=selectInterpolation, inplace=False)

            st.text("Datframe information after data cleaning")
            st.session_state.dftidy = dfCleaned
            dfRawRange = dfCleaned

            dfInfo = pd.DataFrame()
            dfInfo["Types"] = dfRawRange.dtypes
            dfInfo["Missing Values"] = dfRawRange.isnull().sum()
            dfInfo["Missing Values % "] = (
                dfRawRange.isnull().sum()/len(dfRawRange)*100)

            st.table(dfInfo.astype(str))
            st.markdown(" ")
            st.markdown("**Attention**: While the option *'Do data cleaning'*\
                        is selected, the data that will be considered for the next steps of journey at SPPE will be the *cleaned version* of the data")
            st.markdown(" ")