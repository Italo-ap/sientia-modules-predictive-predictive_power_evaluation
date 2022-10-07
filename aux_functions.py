import pandas as pd
from typing import Union
import datetime
import gc
import time
from rich import print as rprint


def check_str(cell) -> bool:
    """
    Function that checks if a given input is a string that cannot be
        converted to a float, e.g. "test" would return True while "1.23" would
        return False.

    Args:

        cell (_type_): _description_

    Returns:

        bool: boolean indicating if input is a real string.
    """

    if isinstance(cell, str):
        try:
            float(cell)
        except ValueError:
            return True
        else:
            return False
    else:
        return False


def mixed_data_cols_rm_str(dataframe: pd.DataFrame, inplace: bool = False,
                           main_type: str = 'float64',
                           return_types: bool = False
                           ) -> Union[pd.DataFrame, None]:
    """
    If a dataframe has columns with mixed datatypes, rows with strings
        are dropped.

    Args:

        dataframe (pd.DataFrame): dataframe with columns with mixede data types
        inplace (bool, optional): if True, the modifications to input dataframe
            are performed inplace. Defaults to False.
        main_type (str, optional): the numerical type other than string.
             Defaults to 'float64'.
        return_types (bool, optional): _description_. Defaults to False.

    Returns:

        pd.DataFrame: the data with string rows removed, if inplace is False.
    """

    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
    mixed_types = []
    mixed_type_cols = [i for i, col in enumerate(df.dtypes)
                       if col != main_type]
    if mixed_type_cols:
        df_mixed = df.iloc[:, mixed_type_cols]
        if return_types:
            mixed_types = df_mixed.applymap(type).apply(pd.unique)
        str_mask = df_mixed.applymap(check_str)
        # If you want to drop rows with strings
        rows_with_str = str_mask.any(axis=1)
        df = df[~rows_with_str]
        df = df.astype(float)
    if not inplace:
        if return_types:
            return df, mixed_types
        else:
            return df
    else:
        dataframe = df
        return None


def get_target_and_inputs(full_dataset: pd.DataFrame, target: str):
    '''
        Extract the input and target variables from the dataset.
        The target timestamp is adjusted according to the particular
        data collection configuration of the anglo plant.

        Args:
            full_dataset (pd.DataFrame): full dataframe with both input vars
                                        and target measurements.
            target (str): string for the column name of the target measurement.
    '''

    inputs_dataset = full_dataset[full_dataset[target].notna()]
    target_series = inputs_dataset[[target]]
    inputs_dataset = inputs_dataset.drop(target, axis=1)

    # Adjustment required due to PIMS interpolation strategy: a newly published
    # value is carried forward until a new sample arrives, even though that new
    # value refers to samples collected in the past interval. The 10 sec
    # sliding makes a newly published value arrive only in the next sample, so
    # that, for instance, a sample at 17:00 is the same as at 16:59:50.
    # Allows you to look back when trying to predict a value at 17:00:00,
    # for thesample at that time is associated with the interval 1
    # 5:00:10-17:00:00. If the 10sec shift is removed, the sample at
    # 17:00:00 would be associated with the interval 17:00:00-18:59:50 --- that
    # would be less intuitive.
    target_series = target_series.shift(-2, freq='H')
    target_series = target_series.shift(10, freq='S')

    return target_series, inputs_dataset


def shutdown_mask(inputs_dataset: pd.DataFrame):

    mask = ((inputs_dataset['Ore Pulp Flow'] == 0.0) | (
        inputs_dataset['Ore Pulp Density'] <= 1.0))

    return mask

def fake_flot_df():
    df = []