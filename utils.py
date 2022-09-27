import streamlit as st
import pandas as pd
import datetime

# ===========================================
# Functions of Sanity Check
# ===========================================


def validate_datetime_index_lastvalue(last_index_value: datetime) -> bool:
    """
        ## Summary
            Check if .
        Args:
            data_string value of datetime
            format -> desired format(s)

        Returns:
            (bool): A bool value of True if datetime string is supported.
    """
    
    try:
        if str(last_index_value) != 'NaT':
            endTimeDf = last_index_value
            correct_last_value_csvfile = True
            print("There is no bug with the last value of the csv file.")
    except ValueError:
        raise ValueError(
            "There must exist a non time date value in thes last row of your csv file")

    return last_index_value


def validate_datetime(date_string, format: str) -> bool:
    """
        ## Summary
            Check if datetime string format is supported and return True if is the correct format.
        Args:
            data_string value of datetime
            format -> desired format(s)

        Returns:
            (bool): A bool value of True if datetime string is supported.
    """
    try:
        datetime.datetime.strptime(date_string, format)
        print("Using the correct date string format.")
        correct_format = True
    except ValueError:
        raise ValueError(
            "This is the incorrect date string format. It should be %Y-%m-%d %H:%M:%S")

    return correct_format

# ===========================================
# Functions of Data Retrieval
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

    date_string = dataFrame.index[0]
    print(date_string)
    print(type(date_string))
    format = "%Y-%m-%d %H:%M:%S"
    correct_format = validate_datetime(date_string, format)

    if correct_format:

        dataFrame.index = pd.to_datetime(
            dataFrame.index, format='%Y-%m-%d %H:%M:%S')
        dataFrame = dataFrame.sort_index(ascending=True)
        dataFrame = dataFrame.apply(pd.to_numeric, errors='coerce')

    return dataFrame
