import pandas as pd
import numpy as np
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(base_dir, 'data')
ff5_file = os.path.join(data_dir, '5_Industry_Portfolios_Daily.csv')
ff10_file = os.path.join(data_dir, '10_Industry_Portfolios_Daily.csv')
ff5_month_file = os.path.join(data_dir, '5_Industry_Portfolio_Monthly.csv')
ff10_month_file = os.path.join(data_dir, '10_Industry_Portfolio_Monthly.csv')

# value weighted 5 industry portfolios
ff5_df = pd.read_csv(ff5_file, index_col=0, header=5, parse_dates=True)
ff5_df = ff5_df.reset_index()
ff5_df = ff5_df.iloc[:25901, :]
ff5_df = ff5_df.rename(columns={'index': 'Date'})
ff5_df['Date'] = pd.to_datetime(ff5_df['Date'], format='%Y%m%d')
ff5_df.iloc[:, 1:] = ff5_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
ff5_df.iloc[:, 1:] = ff5_df.iloc[:, 1:].div(100)

# value weighted 10 industry portfolios
ff10_df = pd.read_csv(ff10_file, index_col=0, header=5, parse_dates=True)
ff10_df = ff10_df.reset_index()
ff10_df = ff10_df.iloc[:25901, :]
ff10_df = ff10_df.rename(columns={'index': 'Date'})
ff10_df['Date'] = pd.to_datetime(ff10_df['Date'], format='%Y%m%d')
ff10_df.iloc[:, 1:] = ff10_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
ff10_df.iloc[:, 1:] = ff10_df.iloc[:, 1:].div(100)

ff5_df_month = pd.read_csv(ff5_month_file, index_col=0, parse_dates=True)
ff10_df_month = pd.read_csv(ff10_month_file, index_col=0, parse_dates=True)
