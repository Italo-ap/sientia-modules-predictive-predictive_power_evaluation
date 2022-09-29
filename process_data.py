from socketserver import ForkingUDPServer
import pandas as pd
import matplotlib.pyplot as plt
# -------------------------------
df = pd.read_csv("MiningProcess_Flotation_Plant_Database.csv",
                 low_memory=False, decimal=",", parse_dates=["date"], infer_datetime_format=True)


def datacleaning(df):

    # get a series of unique hourly timestamps
    hours = pd.Series(df['date'].unique())
    hours.index = hours

    # create a date time index from the first to the last hour included in the date column
    date_range = pd.date_range(
        start=df.iloc[0, 0], end='2017-09-09 23:59:40', freq='20S')
    # remove first couple observations consistent with the counts exploration above
    date_range = date_range[6:]

    # create lists from both the hours series and the new datetime index
    hours_list = hours.index.format()
    seconds_list = date_range.format()

    # match the new datetime index to the hours series and only append the timestamps if the datea and hour match the hours list
    new_index = []
    for idx in seconds_list:
        if (idx[:13] + ':00:00') in hours_list:
            new_index.append(idx)
    # remove the one missing interval within the hour which we found earlier using the counts
    new_index.remove('2017-04-10 00:00:00')
    print(len(new_index))
    print(len(df))

    df['index'] = new_index
    df['index'] = pd.to_datetime(df['index'])
    df.index = df['index']
    df = df.loc[:, df.columns[:-1]]
    df.rename(columns={'date': 'datetime hours'}, inplace=True)

    return df


dftidy = datacleaning(df)
print(type(dftidy))

plt.plot(dftidy.iloc[:, 1])


# We downsample from hourly to 3 day frequency aggregated using mean
dfresampled = df.resample('2h').mean()
print(dfresampled.head(5))
# plt.figure()
plt.plot(dfresampled.index, dfresampled.iloc[:, 0])
plt.show()
