# %%
# Setting up
from queue import Empty
import pandas as pd
import matplotlib.pyplot as plt
from timeseries_functions import *
from aux_functions import shutdown_mask
# -------------------------------

# Custom Time Series cleaning function


def datacleaning(df):
    '''
    Bunch of code lines found on this notebook at Kaggle:
    https://www.kaggle.com/code/mkoerner1/iron-mining-production-prediction
    That clean and transform data
    '''
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


# Loading data
dfraw = pd.read_csv("MiningProcess_Flotation_Plant_Database.csv",
                    low_memory=False, decimal=",", parse_dates=["date"], infer_datetime_format=True)
#plt.plot(dfraw.iloc[:, 1])
# --------------------------------------------------------

# %%
# Running custom datacleaning found on Kaggle
dfcleaned = datacleaning(dfraw)
#plt.plot(dfcleaned.iloc[:, 1])

# %%
# Getting columns list
columns_list = ['% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow',
                'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
                'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow',
                'Flotation Column 03 Air Flow', 'Flotation Column 04 Air Flow',
                'Flotation Column 05 Air Flow', 'Flotation Column 06 Air Flow',
                'Flotation Column 07 Air Flow', 'Flotation Column 01 Level',
                'Flotation Column 02 Level', 'Flotation Column 03 Level',
                'Flotation Column 04 Level', 'Flotation Column 05 Level',
                'Flotation Column 06 Level', 'Flotation Column 07 Level',
                '% Iron Concentrate', '% Silica Concentrate']

# --------------------------------------------------------------
# %%
# Building fake dataframe for test purpose
dffake = dfcleaned.head(5)
print(dffake[columns_list])
dffake['Ore Pulp Flow'][1] = 0.0
dffake['Ore Pulp Density'][1] = 1.0
# print(dffake)
print(dffake.dtypes)
dffake = dffake.astype({"Ore Pulp Density": str}, errors='raise')
print(dffake.dtypes)

# ---------------------------------------------------------------
# %%
# Appllying first ts function that check if there is any mixed columns
dftidy, mixed_types = mixed_data_cols_rm_str(
    dffake[columns_list], inplace=False, return_types=True)
if not mixed_types.empty:
    print('There is mixed types')
plt.plot(dftidy.index, dftidy.iloc[:, 4])
plt.show()
print(mixed_types)
# print(dftidy2.dtypes)
exit()
# %%
print(dftidy2[columns_list])

# %%
# Getting defined shutdown mask
mask = shutdown_mask(dftidy)
check_mask = []
for item in mask:
    if item:
        check_mask.append(item)
print(check_mask)

# Running shutdown samples
dfnoshutdown, shutdown_dict = rm_shutdown_time(dftidy, mask, 0.0, 0.0, False)
plt.plot(dfnoshutdown.index, dfnoshutdown.iloc[:, 0])
plt.show()
print(shutdown_dict)
print(dfnoshutdown)
print(type(dfnoshutdown))

# %%
print(df[df['Ore Pulp Density'] <= 1.0])
print(df[df['Ore Pulp Flow'] <= 0.0])


# %%
dfnostatic = remove_static_windows(dftidy, threshold=10)
# %%
plt.plot(dfnostatic.index, dfnostatic.iloc[:, 0])
plt.show()


# %%

# %%
plt.plot(dfnostatic.index, dfnostatic.iloc[:, 1])
plt.show()
