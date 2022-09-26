import streamlit as st
import pandas as pd
import datetime

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

    date_string = dataFrame.index
    format = "%Y-%m-%d %H:%M:%S"

    try:
        datetime.datetime.strptime(date_string, format)
        print("Using the correct date string format.")
    except ValueError:
        print("This is the incorrect date string format. It should be %Y-%m-%d %H:%M:%S")

    dataFrame.index = pd.to_datetime(
        dataFrame.index, format='%Y-%m-%d %H:%M:%S')
    dataFrame = dataFrame.sort_index(ascending=True)
    dataFrame = dataFrame.apply(pd.to_numeric, errors='coerce')

    return dataFrame
