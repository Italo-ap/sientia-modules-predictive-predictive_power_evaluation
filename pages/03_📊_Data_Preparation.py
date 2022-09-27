# native imports
import datetime

# common imports
import pandas as pd

# special imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#from statsmodels.tsa.seasonal import seasonal_decompose

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


        ################
        ### Describe ###
        ################

        showDescribe = st.checkbox(
            label='Mostrar estatística descritiva', value=False, key='showDescribe')
        if (showDescribe):
            st.table(dfRawRange.describe().transpose())

        ##########################
        ### Análise Individual ###
        ##########################

        showInfoVar = st.checkbox(
            label="Análisar graficamente cada variável (tendência, sazonalidade, histogram e boxplot",
            value=False,
            key='showInfoVar')
        # Lógica IF para fazer aparecer a tendencia e sazonalidade quando a variável não possuir
        # valores nulos. Caso tenha valores nulos é plotado apenas a variavel
        if (showInfoVar):

            st.write("Selecione a váriavel a ser analisada")
            fltPlot = st.selectbox(
                label='', options=state.fltTags, key='fltPlot')

            if state.dfRawRange[fltPlot].isnull().sum() > 0:

                figDecompose = go.Figure()

                figDecompose.add_trace(go.Scatter(x=state.dfRawRange.index,
                                                    y=state.dfRawRange[fltPlot],
                                                    name=fltPlot))

            else:
                st.write(
                    "Análise de decomposição da série temporal (tendência, sazonalidade e ruído)")
                periodDecompose = st.number_input(
                    "Informe o período de amostragem da série temporal",
                    min_value=2,
                    max_value=360,
                    value=2,
                    step=1,
                    format="%i",
                    key="periodDecompose")

                serieDecompose = seasonal_decompose(
                    state.dfRawRange[fltPlot], model='additive', period=periodDecompose)

                figDecompose = make_subplots(
                    rows=4, cols=1, shared_xaxes=True)

                figDecompose.add_trace(go.Scatter(x=state.dfRawRange.index,
                                                    y=state.dfRawRange[fltPlot],
                                                    name=fltPlot),
                                        row=1, col=1)

                figDecompose.add_trace(go.Scatter(x=state.dfRawRange.index,
                                                    y=serieDecompose.trend,
                                                    name="Tendência"),
                                        row=2, col=1)

                figDecompose.add_trace(go.Scatter(x=state.dfRawRange.index,
                                                    y=serieDecompose.seasonal,
                                                    name="Sazonalidade"),
                                        row=3, col=1)

                figDecompose.add_trace(go.Scatter(x=state.dfRawRange.index,
                                                    y=serieDecompose.resid,
                                                    name="Resíduos"),
                                        row=4, col=1)

                figDecompose.update_layout(
                    xaxis2_rangeslider_visible=False,
                    xaxis2_rangeslider_thickness=0.1)

            figDecompose.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=30,
                                    label="30min",
                                    step="minute",
                                    stepmode="backward"),
                            dict(count=1,
                                    label="1h",
                                    step="hour",
                                    stepmode="backward"),
                            dict(count=1,
                                    label="1d",
                                    step="day",
                                    stepmode="backward"),
                            dict(count=7,
                                    label="1w",
                                    step="day",
                                    stepmode="backward"),
                            dict(count=1,
                                    label="1m",
                                    step="month",
                                    stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
            )
            st.plotly_chart(figDecompose)

            st.info(
                "Para cada gráfico de tendência, sazonalidade e ruído, se ocorrerem valores maiores que zero,\
                significa que a série pode ter algum desses fatores e, nesse caso,\
                deve-se avaliar mais a fundo técnicas para compensar tais fatores.",
            )

            st.write("Histograma e Boxplot")
            figHist = px.histogram(
                state.dfRawRange, x=fltPlot, marginal="box")
            st.plotly_chart(figHist)