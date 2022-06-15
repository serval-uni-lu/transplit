import requests
import os
from io import BytesIO
from zipfile import ZipFile
import pandas as pd

if __name__ == "__main__":
    print("From https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014")
    print("Downloading data (250Mo)...")
    res = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip')
    print("Processing it...")

    with ZipFile(BytesIO(res.content)) as zip:
        df = pd.read_csv(zip.open("LD2011_2014.txt"), sep=';', decimal=',')
        # selecting the first 321 clients in the years 2012, 2013, 2014
        # (starting at 0:00, ending at 23:45)
        df = df.iloc[365*24*4-1:-1, 1:322]
        # summing values to get hourly consumptions
        arr = df.values.reshape(-1, 4, 321).sum(axis=1)
        df2 = pd.DataFrame(arr, columns=df.columns)
        df2 = df2.round() / 1000
        # adding date
        df2.insert(0, 'date', pd.date_range(
            start="2012-01-01",
            end="2015-01-01",
            freq='1H',
            closed="left"
        ))
        # saving to csv
        os.makedirs('dataset/electricity', exist_ok=True)
        df2.to_csv('dataset/electricity/electricity.csv', index=False, float_format='%.3f')
        print("Saved to ./dataset/electricity/electricity.csv")
