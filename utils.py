import streamlit as st
import pandas as pd


# ===========================================
# Functions Data Retrieval
# ===========================================

@st.cache(allow_output_mutation=True)
def getDataFromCSV(file, separator_format: str, decimal_format: str) -> pd.DataFrame:
    """
        ## Summary
            Parse csv file with time series data in BR regional formant and return data in DataFrame format.
        Args:
            csv file  -> which contains the data in CSV format
            separator -> data separator format string used in the csv file
            decimal   -> data decimal format string used in the csv file
        Returns:
            (dataframe): A pd.DataFrame structured data format.
    """
    dataFrame = pd.read_csv(file, sep=separator_format, decimal=decimal_format,
                            encoding="UTF-8",
                            index_col=0,
                            low_memory=False)

    dataFrame.index = pd.to_datetime(
        dataFrame.index, format='%Y-%m-%d %H:%M:%S')
    dataFrame = dataFrame.sort_index(ascending=True)
    dataFrame = dataFrame.apply(pd.to_numeric, errors='coerce')
    
    return dataFrame