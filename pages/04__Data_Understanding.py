# native imports
import datetime

# common imports
import pandas as pd
from rich import print as rprint

# special imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_profiling

from statsmodels.tsa.seasonal import seasonal_decompose

# front-end imports
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

# ===========================================
# Setup
# ===========================================

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
dfRawRange = st.session_state.dftidy
fltTags = st.session_state.filteredTags


##########################
### App page beginning ###
##########################
st.title('Data Understanding - UNDER CONSTRUCTION')
st.subheader("Now it is time to select and prepare your data, like filtering, cleaning and transforming.")

#####################
### Describe data ###
#####################

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
        label='', options=fltTags, key='fltPlot')

    if dfRawRange[fltPlot].isnull().sum() > 0:

        figDecompose = go.Figure()

        figDecompose.add_trace(go.Scatter(x=dfRawRange.index,
                                            y=dfRawRange[fltPlot],
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
            dfRawRange[fltPlot], model='additive', period=periodDecompose)

        figDecompose = make_subplots(
            rows=4, cols=1, shared_xaxes=True)

        figDecompose.add_trace(go.Scatter(x=dfRawRange.index,
                                            y=dfRawRange[fltPlot],
                                            name=fltPlot),
                                row=1, col=1)

        figDecompose.add_trace(go.Scatter(x=dfRawRange.index,
                                            y=serieDecompose.trend,
                                            name="Tendência"),
                                row=2, col=1)

        figDecompose.add_trace(go.Scatter(x=dfRawRange.index,
                                            y=serieDecompose.seasonal,
                                            name="Sazonalidade"),
                                row=3, col=1)

        figDecompose.add_trace(go.Scatter(x=dfRawRange.index,
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
        dfRawRange, x=fltPlot, marginal="box")
    st.plotly_chart(figHist)

# Pandas Profiling report generation
#pr = dfRawRange .profile_report()
#st.title("Pandas Profiling in Streamlit")
#st.write(dfRawRange )
#st_profile_report(pr)