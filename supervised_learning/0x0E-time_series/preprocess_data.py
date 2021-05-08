#!/usr/bin/env python3
"""
Preprocess raw dataset
"""
import pandas as pd
import matplotlib.pyplot as plt


# making data frame from csv file
data = \
    pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

data = data[0::60]
# creating bool series True for Not NaN values
bool_series = pd.notnull(data["Open"])

# displaying data only with Open = Not NaN
new_data = data[bool_series]
converted_df = pd.to_datetime(new_data["Timestamp"], unit="s")
# new_data = new_data[convert]
print(converted_df)
