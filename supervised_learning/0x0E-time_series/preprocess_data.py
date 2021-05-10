#!/usr/bin/env python3
"""
Preprocess raw dataset
"""
import pandas as pd
import matplotlib.pyplot as plt


# making data frame from csv file
data = \
    pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

data = data.drop(["Open", "High", "Low", "Volume_(BTC)",
                 "Volume_(Currency)", "Weighted_Price"], axis=1)

df = data.dropna()

df = df[0::60]

df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df.reset_index(inplace=True, drop=True)
print(df)
