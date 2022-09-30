import streamlit as st
import pandas as pd
import datetime
from typing import Union, Tuple, Generator
import datetime
import numpy as np
import gc
from sklearn.preprocessing import MinMaxScaler

# ===========================================
# Functions of Data Cleaning and Transformation
# ===========================================


def pre_process(dataframe: pd.DataFrame, window_size: Union[int, str],
                stride: Union[int, str], format_df_start: bool = False,
                hour_coer: bool = False) -> pd.DataFrame:
    """
    Function that divides timeseries in windows of specified size and
        stride. Useful whenever different windowed measurements can be directly
        associated. If necessary, each window can have a specific formatted
        start with respect to its starting hour, second and minute.

        The returned dataframe will have an extra "continuity" column, which
        identifies the samples lying on the edges of continuous windows, i.e.,
        it is possible to collect "window_size" samples before (and including)
        that instant.

    Args:

        dataframe: (pandas.DataFrame) data to be treated - WITH DATETIME INDEX;
        window_size: (int, str) size of the window, in number of samples -
                    must be > 0; may also be a string, such as '2h' for two
                    hour long windows.
        stride: (int, str) size of the step per sample, in number of samples
                - must be > 0; may also be a string, such as '2h' for two
                    hour long steps.
        format_df_start: (bool) if true, all sub-dataframes start with the same
                        minute and second;
        hour_coer: (bool) if true, keeps every initial hour even if the first
                    window is even or odd if the first window starts at an odd
                    hour;
    Returns:

        pd.DataFrame: The ready-to-go dataframe with an extra column named
        'continuity' which indicates the samples that lie at the end of the
        desired windows, taking stride into consideration.
    """

    # Infer the frequency from data.
    data = dataframe.copy()
    freq = pd.infer_freq(data.index[:5])
    t_s = pd.to_timedelta(freq)
    # Get the digits in the frequency, e.g., if the frequency is 5S, freq_digit
    # would be 5.
    freq_digit = int(''.join(filter(str.isdigit, freq)))
    # Due to pandas data treatment, if the sampling time is 1S, the infer_freq
    # return is 'S', so we don't have the digit '1'.
    if not freq_digit:
        freq_digit = 1
    # Get the chars in the frequency, e.g., if the frequency is 5S, freq_alpha
    # would be S.
    freq_alpha = str(''.join(filter(str.isalpha, freq)))

    # If the caller specified the window size with a string, we convert it into
    # an equivalent number of samples.
    if isinstance(window_size, str):
        # Remove a single sampling time from the window size, so that the
        # "delta" is whats added to the timestamp of a sample to get the next
        # sample. E.g., 2h-10s= 1h59min50s, which is added to 01h00min10s to
        # get 03h00min00s. The window can't be a closed interval in both sides,
        # so you essentially remove one sample to open the interval in one tip.

        delta = pd.Timedelta(window_size) - pd.Timedelta(freq_digit,  # window
                                                         unit=freq_alpha)
        end_aux = pd.to_datetime('11/15/1996') + delta
        window_size = len(pd.date_range(start='11/15/1996', end=end_aux,
                                        freq=freq))  # Window size in samples.

    if isinstance(stride, str):
        delta = pd.Timedelta(stride) - pd.Timedelta(freq_digit,  # The stride
                                                    unit=freq_alpha)
        end_aux = pd.to_datetime('11/15/1996') + delta
        stride = len(pd.date_range(start='11/15/1996', end=end_aux,
                                   freq=freq))  # Stride in samples.

    # Find the points where the data shows temporal discontinuity.
    # This array will be true in samples where the change in timestamp was
    # greater than what is expected with a regular sampling time of t_s.
    discontinuity = data.index.to_series().diff().gt(t_s).astype(int)
    dis_points = list(np.where(discontinuity == 1)[0]) + [data.shape[0]]

    # The extra column indicating continuity; initially all zeros.
    y = pd.Series(data=np.zeros(data.shape[0]), index=data.index,
                  name='continuity')
    data = data.join(y)

    # Split the dataset into mini, continuous, datasets; put 'em all on a list.
    # ...divide and conquer...
    last_check = 0
    dfs = []
    for ind in dis_points:
        if format_df_start:
            try:
                data_chunk = format_start(
                    data.iloc[last_check:ind],
                    data.index[0].second,
                    data.index[0].minute,
                    data.index[0].hour % 2 if hour_coer else -1)
                if data_chunk.shape[0] >= window_size:
                    dfs.append(data_chunk)
            # This is not an error, it's a signal that we don't have enough
            # data
            except ValueError:
                pass
        else:
            data_chunk = data.iloc[last_check:ind]
            if data_chunk.shape[0] >= window_size:
                dfs.append(data_chunk)
        last_check = ind
    del data
    gc.collect()
    for df in dfs:
        if (df.shape[0] >= window_size):
            # Marking the end of all valid windows.
            df.iloc[(window_size-1)::stride, -1] = 1

    # Turn it all into a dataframe
    x_train = pd.concat(dfs, axis=0)

    return x_train


def format_start(df: pd.DataFrame, s: int = 0, m: int = 0, h: int = -1):
    """
    Function to remove (if necessary) the first rows of the data in
        order to have the first row in a specific format.

    Args:

        df (pd.DataFrame): df whose index is in datetime format.
        s (int): initial second, with all windows starting with this second.
        m (int): initial minute, with all windows starting with this minute.
        h (int): if 1, initial hours are even; id 0, initial hours are odd.

    Returns:

        pd.DataFrame: with the specified format.        

    """

    # leading edge
    if s != -1:
        inis = df.index[0].second
        if inis != s:
            if s < inis:
                delta = pd.Timedelta(str((60+s) - inis)+'s')
            else:
                delta = pd.Timedelta(str((s) - inis)+'s')
            if df.index[0] + delta >= df.index[-1]:
                df = df.loc[(df.index[0]+delta):]
            else:
                raise ValueError("Not enough data!")
    if m != -1:
        inim = df.index[0].minute
        if inim != m:
            if m < inim:
                delta = pd.Timedelta(str(((60+m)-inim))+'m')
            else:
                delta = pd.Timedelta(str(((m)-inim))+'m')
            if df.index[0] + delta >= df.index[-1]:
                df = df.loc[(df.index[0]+delta):]
            else:
                raise ValueError("Not enough data!")
    if h != -1:
        inih = df.index[0].hour
        if inih % 2 != h:
            delta = pd.Timedelta('1h')
            if df.index[0] + delta >= df.index[-1]:
                df = df.loc[(df.index[0]+delta):]
            else:
                raise ValueError("Not enough data!")

    return df


def rm_shutdown_time(data: pd.DataFrame, rm_events_mask: np.ndarray,
                     stop_interval_start: float, stop_interval_end: float,
                     return_shutdown_dict: bool = False,
                     ) -> pd.DataFrame:
    """
    Remove all samples in rm_events_mask, plus/minus stop_interval.

    Args:

        data (pd.DataFrame): data to be processed, must have datetime indexing;
        rm_events_mask (ndarray): boolean ndarray with length equal to number
                             of rows in data, where rows to be removed are True
        stop_interval_start (float): number of hours to be removed before the
                              events in the mask
        stop_interval_end (float): : number of hours to be removed after the
                              events in the mask
        return_shutdown_dict (bool): if True, returns a dictionary with start
                                    and end indices of all events

    Returns:

        pd.DataFrame: data with the rows of rm_events_mask and samples around
            stop_interval removed, in addition to (optionally) shutdown dict.
    """

    dataset = data.copy()
    # All time instants where events are happening
    rm_events_idx = dataset[rm_events_mask].index
    freq = pd.infer_freq(dataset.index[:10])
    t_s = pd.to_timedelta(freq)

    # All time instants where events are happening (in the index), with an
    # indicator (series value equal to 1) of whether that instant is the start
    # (first sample) of an event window.
    t = rm_events_idx.to_series().diff().gt(t_s).astype(int)
    # Index of TIMELINE ARRAY "t" where each event window begins
    event_starts = np.r_[0, np.where(t == 1)[0]]
    # Index of TIMELINE ARRAY "t" where each event window ends
    # Subtract one because the last event ends on the last sample before the
    # next event starts. Add the last sample "t.shape[0]-1" as the end of the
    # last event.
    event_ends = np.r_[np.where(t == 1)[0]-1, t.shape[0]-1]

    shutdown_dicts = []  # List of dicts containing start and end of all events
    for start, end in zip(event_starts, event_ends):
        start_time = t.index[start]
        end_time = t.index[end]
        shutdown_dicts.append({"start": start_time, "end": end_time})
    # All indexes to be removed
    stop_idx = np.empty((0), dtype='datetime64[ns]')

    for event in shutdown_dicts:
        # Starting moment of the window pre-event
        start = event['start'] - datetime.timedelta(
            hours=stop_interval_start)
        # Ending moment of the window post event
        end = event['end'] + datetime.timedelta(
            hours=stop_interval_end)
        interval = pd.date_range(start, end, freq=freq)
        stop_idx = np.r_[stop_idx, np.array(interval)]
        stop_idx = np.unique(stop_idx)

    stop_idx = np.intersect1d(stop_idx, dataset.index)
    dataset = dataset.drop(stop_idx)
    if return_shutdown_dict:
        return dataset, shutdown_dicts
    else:
        return dataset


def summarize_timeseries(inputs_dataset: pd.DataFrame, target: pd.DataFrame,
                         window: Union[pd.DateOffset, pd.Timedelta, str],
                         origin: Union[pd.Timestamp, str] = 'start_day',
                         offset: Union[pd.Timedelta, str] = None,
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize timeseries data split in inputs_dataset (typically a
            dataframe), target (typically a series).

        The indexes of the returned dataframes will match exactly.

        This function is useful when developing predictive models that have
            inputs_dataset as input and target as output.

        The input series will be summarized by its median, while the target
            series is summarized by the last sample.

    Args:

        inputs_dataset (pd.DataFrame): dataframe with input variables
        target (pd.DataFrame): dataframe with target variable
        window (Union[pd.DateOffset, pd.Timedelta, str]): window length over
            which the summarization operation is taken. Will also be the new
            sampling frequency of the dataset.
        origin (Union[pd.Timestamp, str], optional):  the timestamp on which to
         adjust the grouping. The timezone of origin must match the timezone of
         the index. If string, must be one of the following:

            epoch: origin is 1970-01-01
            start: origin is the first value of the timeseries
            start_day: origin is the first day at midnight of the timeseries
            end: origin is the last value of the timeseries
            end_day: origin is the ceiling midnight of the last day

            Defaults to 'start_day'.
        offset (Union[pd.Timedelta, str], optional): an offset timedelta added
            to the origin. Defaults to None.

    Returns:

        Tuple[pd.DataFrame, pd.DataFrame]: inputs_data_resampled,
                                           target_resampled
    """

    inputs_dataset_resampled = inputs_dataset.resample(window,
                                                       origin=origin,
                                                       offset=offset,
                                                       label='right',
                                                       closed='right'
                                                       ).median()

    target_resampled = target.resample(window,
                                       origin=origin,
                                       offset=offset,
                                       label='right',
                                       closed='right'
                                       ).last()

    target_resampled = target_resampled.dropna()

    inputs_dataset_resampled = inputs_dataset_resampled.dropna()
    timeline = np.intersect1d(target_resampled.index,
                              inputs_dataset_resampled.index)
    target_resampled = target_resampled.loc[timeline]
    inputs_dataset_resampled = inputs_dataset_resampled.loc[timeline]

    return (inputs_dataset_resampled, target_resampled)


def add_ar_input(data: pd.DataFrame, target: pd.DataFrame,
                 delay: float = None) -> pd.DataFrame:
    """
    Gets a dataframe and a target series and add the delayed series to the
        dataframe.

    Args:

        data (pd.DataFrame): dataframe with the inputs to the regression model
        target (pd.DataFrame): target series
        delay (float, optional): the delay in seconds to applied to the series.
            If not given, the series is just shifted by a single sample.
            Defaults to None.

    Returns:

        pd.DataFrame: dataframe with the new feature of the delayed target.
    """

    target_name = target.columns[0]
    if delay:
        delayed_target = target.shift(delay, freq='S').rename(columns={
            target_name: target_name+'_delayed'
        })
    else:
        delayed_target = target.shift(1).rename(columns={
            target_name: target_name+'_delayed'
        })
    augmented_data = data.join(delayed_target, how='inner')

    return augmented_data


def load_train_test_scaled(path_train: str, target_name: str,
                           ar: bool = False, path_test: str = '',
                           return_scaler: bool = False,
                           ar_scaling: bool = True
                           ) -> Union[Tuple[pd.DataFrame], dict]:
    """
    Function to load train and test data from files, with scaling of the
        inputs. If selected, the autoregressive input can
        be dropped.

    Args:

        path_train (str): path to the training data csv file.
        target_name (str): column name of the target variable.
        ar (bool, optional): flag that indicates if the autoregressive input
            should be kept. Defaults to False.
        path_test (str, optional): path to the test data csv file.
            Defaults to ''.
        return_scaler (bool, optional): flag to return the MinMaxScaler used.
            Defaults to False.
        ar_scaling (bool): flag that indicates if the autoregressive input
            should be included in the scaling process. Defaults to True.

    Returns:

        Union[Tuple[pd.DataFrame], dict]: tuple with training and test inputs
            and outputs or dictionary with said tuple and data scaler.
    """

    data_train = pd.read_csv(path_train,
                             index_col=0)

    data_test = pd.read_csv(path_test,
                            index_col=0)

    x_train = data_train.drop([target_name], axis=1)
    y_train = data_train[[target_name]]
    delayed_target = target_name + '_delayed'
    if not ar:
        x_train = x_train.drop(delayed_target, axis=1)

    x_test = data_test.drop([target_name], axis=1)
    y_test = data_test[[target_name]]

    if not ar:
        x_test = x_test.drop(delayed_target, axis=1)
    del [[data_train, data_test]]
    gc.collect()

    if not ar_scaling:
        y_train_delayed = x_train[delayed_target]
        y_test_delayed = x_test[delayed_target]

        x_train.drop([delayed_target], axis=1, inplace=True)
        x_test.drop([delayed_target], axis=1, inplace=True)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(data=x_train_scaled, index=x_train.index,
                           columns=x_train.columns)

    x_test_scaled = scaler.transform(x_test)
    x_test = pd.DataFrame(data=x_test_scaled, index=x_test.index,
                          columns=x_test.columns)

    if not ar_scaling:
        x_train = x_train.join(y_train_delayed)
        x_test = x_test.join(y_test_delayed)

    if return_scaler:
        return {'data': (x_train, y_train, x_test, y_test),
                'scaler': scaler}
    else:
        return (x_train, y_train, x_test, y_test)


def counts_ratio_per_batch(timeseries: pd.Series, batches_dicts: list,
                           column: str) -> pd.DataFrame:
    """
    Calculates the ratio between counts of all values in a batch and the
        samples in the batch. Used for series with discrete inputs as a way of
        aggregating a batch.

    Args:

        timeseries (pd.Series): The dataset with the column
        batches_dicts (list): The list containing the start and end of
            every batch
        column (str): The column name

    Returns:

        df_column_values (pd.DataFrame): A dataframe contaning the ratio of
        column in every data window.
    """

    # Possible values for that column in any batch
    values = list(timeseries.unique())

    # The final list with every ratio value for every batch
    column_values = []

    for event in batches_dicts:
        # Starting moment of the window
        start = event['start']
        # Ending moment of the window
        end = event['end']

        # Create a key for every possible value in that batch
        batch_values = {value: 0 for value in values}

        # Count of each value in that batch
        values_event = timeseries.loc[start:end].value_counts().to_dict()
        size_event = timeseries.loc[start:end].shape[0]

        # * Since the batch does not necessarily has all the values, we use
        # * it's value counts to update the batches values
        for key, _ in values_event.items():
            batch_values[key] = values_event[key]/size_event

        column_values.append(batch_values)

    df_column_values = pd.DataFrame(column_values)

    for df_column in df_column_values.columns:
        df_column_values = df_column_values.rename(
            {
                df_column: f'{column}_{df_column}'
            },
            axis='columns')

    # Here we drop the first column due to redundancy. The value of the first
    # column is always 1 minus the sum of the others.
    df_column_values.drop(df_column_values.columns[0], axis=1)
    return df_column_values


def crosscorr(series_a: pd.DataFrame, series_b: pd.DataFrame,
              lag: int = 0, diff: bool = True) -> pd.DataFrame:
    """
    Get the correlation from two series with different lags between them.

    Args:

        series_a: (pandas.DataFrame) input data to correlate with the target;
        series_b: (pandas.DataFrame) target timeseries data;
        lag: (int) number of shifts in input data to explore correlation;
        diff: (bool) if true, takes the first difference series of the
            timeseries.

    Returns:
        A DataFrame with the Best Lag and Correlation value for each input
            variable
    """
    series_a_copy = pd.DataFrame(series_a.copy())
    series_b_copy = pd.DataFrame(series_b.copy())
    lags = range(lag)
    cols = pd.DataFrame({'Best lag': np.zeros(len(series_a_copy.columns)),
                         'Correlation': np.zeros(len(series_a_copy.columns))},
                        index=list(series_a_copy.columns))
    if diff:
        series_b_copy = pd.Series(np.ediff1d(series_b_copy, to_begin=0),
                                  series_b_copy.index)

    for col in cols.index:
        if diff:
            series_a_copy[col] = np.ediff1d(series_a_copy[col], to_begin=0)
        for lag in lags:
            candidate = series_a_copy[col].shift(lag).corr(series_b_copy)
            if abs(candidate) > abs(cols['Correlation'].loc[col]):
                cols['Correlation'].loc[col] = candidate
                cols['Best lag'].loc[col] = lag
    return cols


def remove_static_windows(data: pd.DataFrame, columns: list = None,
                          threshold: int = 10) -> pd.DataFrame:
    """
    Remove windows where there is no variation for 'threshold' consecutive
        samples. In other words, remove static windows, where a window is
        considered static if the value of the series remains unchanged for
        at least 'threshold' samples.

    Args:
        data (pd.DataFrame): Dataframe with the values to be analyzed
        columns (list, optional): Columns to check if the windows are static.
            Defaults to [].
        threshold (int, optional): Minimum length of sequence of samples with
            the same value to remove. Defaults to 10.

    Returns:
        pd.DataFrame: Dataframe without the static windows.
    """
    if not columns:
        columns = data.columns
    for column in columns:
        series = data[column]
        series_diff = series.diff()
        series_diff_0 = np.where(series_diff == 0)[0]
        if list(series_diff_0):
            dis_points_start = [series_diff_0[0]] + list(series_diff_0[
                np.where(np.diff(series_diff_0,
                         prepend=series_diff_0[0]) > 1)[0]])
            dis_points_end = (list(series_diff_0[np.where(np.diff(
                series_diff_0, prepend=series_diff_0[0]) > 1)[0] - 1]) +
                [series_diff_0[-1]])

            for start, end in zip(dis_points_start, dis_points_end):
                if end - start >= threshold:
                    # * Drop including the 'end' sample
                    data = data.drop(data[start:end+1].index)

    return data


def get_sample_time(data: pd.DataFrame) -> float:
    """
    Infers sample time from data

    Args:

        data (pd.DataFrame): Data to infer the sample time

    Returns:

        t_s (float): The inferred sample time
    """

    freq = pd.infer_freq(data.index[:10])
    sample = 1
    while freq is None:
        freq = pd.infer_freq(data.index[sample: 10 + sample])
        sample += 1

    if str(freq) == 'S':
        t_s = 1
    else:
        t_s = pd.to_timedelta(freq)

    return t_s


def get_batches_dict(data: pd.DataFrame, t_s, min_time: str,
                     cut_head: int = 0, cut_tail: int = 0):
    """
    Get the start and end timestamps for each active batch present in the
    dataset. Used in the dataset after removing the shutdown time, so we only
    have discontinuity when the batch changes, but all the data is when the
    process is active. Furthermore, since the time for each batch is different,
    get to know the start and end of each one is extremely important.


    Args:

        data (pd.DataFrame): The data to obtain the start and end timestamps

    Returns:

        batches_dicts (list): The list of timestamps
    """
    dataset = data.copy()

    # All time instants where events are happening, with an indicator of
    # whether that instant is the start (first sample) of an event window.

    # ! Since we're using the diff from numpy, we have to prepend the first
    # ! value because the function does not return NaN for the first sample
    # ! as the diff from pandas.

    t = dataset.index.to_series().diff().gt(t_s).astype(int)

    # Index of timeline array "t" where each event window begins
    event_starts = np.r_[0, np.where(t != 0)[0]]
    # Index of timeline array "t" where each event window ends
    # Subtract one because the last event ends on the last sample before the
    # next event starts. Add the last sample "t.shape[0]-1" as the end of the
    # last event.

    event_ends = np.r_[np.where(t != 0)[0] - 1, t.shape[0] - 1]
    batches_dicts_time = []
    batches_dicts_index = []

    # Dict containing start and end of all events
    # Remove samples in the head and tail of the series
    event_starts += cut_head
    event_ends -= cut_tail
    for start, end in zip(event_starts, event_ends):
        if (end > start):
            start_time = dataset.index[start]
            end_time = dataset.index[end]
            if end_time - start_time > pd.to_timedelta(min_time):
                batches_dicts_time.append(
                    {"start": start_time, "end": end_time})
                batches_dicts_index.append({"start": start, "end": end})
    return batches_dicts_time, batches_dicts_index


def get_dfs_from_batches(data: pd.DataFrame, batches_dicts: list):
    """
    Get the dataframes for each batch present in the dataset.

    Args:

        data (pd.DataFrame): The data to obtain the start and end timestamps
        batches_dicts (list): The list of timestamps

    Returns:

        dfs (list): The list of dataframes

    """
    dfs = []
    for batch in batches_dicts:
        dfs.append(data.loc[batch["start"]:batch["end"]])

    return dfs


def generate_ar_process(coefs: list, length: int = 100):
    """
    Generate synthetic data from an Autoregressive (AR) process of a given
    length and known coefficients.

    Args:
        coefs (list): Array-like with coefficients of lagged measurements of
        the series.
        length (int): number of data points to be generated.

    Returns:
        series: array with the generated series
    """

    order = len(coefs)
    coefs = np.array(coefs)

    y = np.zeros(length)
    # Initial values y[0, 1, .., order]
    y[:order] = [np.random.normal() for _ in range(order)]

    for k in range(order, length):
        # Get previous values of the series, reversed
        prev_vals = y[(k-order):k][::-1]

        y[k] = np.sum(np.array(prev_vals) * coefs) + np.random.normal()

    return np.array(y)


def local_global_test(series, stat_func, window_length, stride):
    """
    Calculate statistics for a timeseries globally and locally with windows of
    a set length and stride.

    Args:
        series (array): Array with timeseries measurements
        stat_func (function): Function that calculates a statistic given an
        array of measurements
        window_length (int): samples in a window for calculating statistics
        stride (int): step size between different windows

    Returns:
        stats_dict: dictionary with statistic for the global series and array
        of statistics for each window
    """
    local_stats = []
    global_stat = stat_func(series)
    bool_indices = np.zeros(len(series))
    # Indicator array with the last sample on each valid window. If position
    # k of the array is 1, that means that 'window_length' samples before and
    # including that sample can be taken to construct a window
    bool_indices[(window_length-1)::stride] = 1
    # n_windows = np.sum(bool_indices)
    window_indices = np.where(bool_indices == 1)[0]
    for idx in window_indices:
        # Plus one because the last sample in 'idx' should be included in the
        # window
        window = series[(idx-window_length+1):(idx+1)]
        local_stats.append(stat_func(window))

    return {'global': global_stat, 'locals': local_stats}

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
