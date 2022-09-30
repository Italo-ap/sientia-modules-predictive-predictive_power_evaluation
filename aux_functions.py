import pandas as pd
from typing import Union
import datetime
import gc
import time
import datatable as dt
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

def load_data(path: str, test_th: str = '', batch_path: str = '',
              return_test: bool = False):
    '''
        Load the data to a dataframe and select the timestamp as datetime type
        index. Batch data is provided in a different file, which is also loaded
        and merged in the returned dataframe.

        Args:
            path (str): path to the main csv file, with the timestamp as its
                        first column.
            test_th (str): timestamp that divides train and test sets. All data
                           collected after this time will be discarded.
            batch_path (str): path to the batch data csv file, with the
                              timestamp as its first column.
        Returns:
            full_dataset (pd.DataFrame): dataframe with both the process and
                                        batch measurements.
    '''

    full_dataset = dt.fread(path)
    full_dataset.key = 'Timestamp'
    full_dataset = full_dataset.to_pandas()

    full_dataset.index = pd.to_datetime(full_dataset.index,
                                        format='%Y/%m/%d %H:%M:%S')
    full_dataset = full_dataset.sort_index(ascending=True)
    # Data was collected in UTC, we convert it to UTC-3
    full_dataset.index = full_dataset.index - datetime.timedelta(hours=3)
    #full_dataset = add_batch_data(batch_path, full_dataset)
    if test_th:
        test_dataset = full_dataset.loc[pd.to_datetime(test_th):]
        full_dataset = full_dataset.loc[:pd.to_datetime(test_th)]
    if return_test:
        return full_dataset, test_dataset
    else:
        return full_dataset


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

    mask = ((inputs_dataset['FIT0413_610__PT__VAL_PV_OUT'] > 0.0) | (
        inputs_dataset['FIT0413_008__PT__VAL_PV_OUT'] > 0.0)) & \
        (inputs_dataset['DIT0413_003B__PT__VAL_PV_OUT'] >= 1.3)

    return mask
