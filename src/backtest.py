# %%
import os
import sys
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from optimize import sharpe
from optimize import oos_pbcv
from optimize import cvar_opt, cvar_pbr_opt, cvar_relax_opt, u_min_cvar
from optimize import mv_opt, mv_pbr1_opt, mv_pbr2_opt, u_min_mv1, u_min_mv2
from data_fetch import ff5_df, ff10_df
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=200)

# %%
columns_to_convert = ff5_df.columns[1:]
ff5_df[columns_to_convert] = ff5_df[columns_to_convert].astype(float)
columns_to_convert = ff10_df.columns[1:]
ff10_df[columns_to_convert] = ff10_df[columns_to_convert].astype(float)

# %%
# %%
# Mean-Variance Optimization
# 1. MV-SAA
# 2. MV-PBR-R1
# 3. MV-PBR-PSD
# 4. MV-SAA with no short selling
# 5. L1 regularization ||w||_1 <= Um
# 6. L2 regularization ||w||_2 <= Um
# 7. Minimum variance ...
# Benchmark: equal weights

# 1. CV-SAA
# 2. CV-PBR with U2 = infinity
# 3. CV-PBR with U1 = infinity
# 4. CV-SAA with U1 and U2 Calibrated using OOS-PBCV


pbr_funcs = {
    'mv_saa': mv_opt,  # benchmark
    # 'mv_pbr1': mv_pbr1_opt,
    # 'mv_pbr2': mv_pbr2_opt,
    'cvar_saa': cvar_opt,  # benchmark
    'cvar_pbr': cvar_pbr_opt,
    'cvar_pbr': cvar_relax_opt,
}


class Backtester:
    def __init__(self, data: pd.DataFrame,
                 lookback: int,
                 rebalance_freq: int,
                 R_target: float,
                 start_date: datetime,
                 end_date: datetime,
                 U_tuning_params: dict = None):

        self.data = data.copy()
        self.lookback = lookback
        self.R_target = R_target
        self.rebalance_freq = rebalance_freq
        self.start_date, self.end_date = start_date, end_date

        start_index = data[data['Date'] >= start_date].index[0]
        start_index = start_index - lookback + 1
        end_index = data[data['Date'] <= end_date].index[-1]
        self.backtest_data = data.iloc[start_index:end_index +
                                       1].reset_index(drop=True)

        self.num_assets = len(self.backtest_data.columns) - 1
        self.num_days = len(self.backtest_data) - lookback + 1
        self.dates = self.backtest_data['Date'].iloc[lookback - 1:].to_numpy()

        # these params are hyperparameters used for tuning U
        self.alpha = U_tuning_params['alpha']
        self.beta = U_tuning_params['beta']
        self.div = U_tuning_params['div']
        self.bit = U_tuning_params['bit']
        self.gamma = U_tuning_params['gamma']
        self.k_fold = U_tuning_params['k']

        self.strategies = [
            'equal',
            'mv',
            'mv_pbr1',
            'cvar',
            'cvar_pbr'
        ]

        self.curr_weights = {
            'equal': np.zeros(self.num_assets),
            'mv': np.zeros(self.num_assets),
            'mv_pbr1': np.zeros(self.num_assets),
            'cvar': np.zeros(self.num_assets),
            'cvar_pbr': np.zeros(self.num_assets)}

        self.weights_records = {
            'mv': [np.zeros(self.num_assets)],
            'mv_pbr1': [np.zeros(self.num_assets)],
            'cvar': [np.zeros(self.num_assets)],
            'cvar_pbr': [np.zeros(self.num_assets)]}

        self.u_records = {
            'mv_pbr1': [],
            'cvar_pbr': []}

        self.navs = {
            'equal': [1],
            'mv': [1],
            'mv_pbr1': [1],
            'cvar': [1],
            'cvar_pbr': [1]}

        self.max_tries = 5

    def run(self):
        # adjust position
        # note: At re-balance date:
        # 1. Calculate return for the last re-balance period,
        # 2. Re-balance
        current_adjust = 0
        total_days = len(self.backtest_data)
        total_adjust = total_days // self.rebalance_freq

        adjust_count = 0
        for i in range(self.lookback - 1, len(self.backtest_data)):
            row = self.backtest_data.iloc[i]
            today_ret = row.iloc[1:].to_numpy()
            today_date = row.iloc[0]
            # update nav
            for stra, w in self.curr_weights.items():
                self.navs[stra].append(
                    self.navs[stra][-1] * (1 + np.sum(today_ret * w)))

            # update weights
            if adjust_count % self.rebalance_freq == 0:
                current_adjust += 1
                adjust_count = 0
                print(
                    f"\n{'#'*80}\nRe-balance at date: {today_date} | {current_adjust}/{total_adjust}")
                window = self.backtest_data.iloc[i - self.lookback + 1: i + 1]
                X = window.iloc[:, 1:].to_numpy()
                mu = np.mean(X, axis=0)
                Sigma = np.cov(X, rowvar=False, ddof=1)

                # record w_equal
                self.curr_weights['equal'] = np.ones(
                    self.num_assets) / self.num_assets

                print(f"\tRe-balance mv...")
                for i in range(self.max_tries):
                    try:
                        w_mv = mv_opt(mu, Sigma, self.R_target)
                        w_mv = np.array(w_mv)
                        self.curr_weights['mv'] = w_mv
                        self.weights_records['mv'].append(w_mv)
                        break
                    except Exception as e:
                        if i < self.max_tries - 1:
                            print(f"\t\t{e}")
                            print(f"\tRetrying... | {i+1}/{self.max_tries}")
                        else:
                            print(f"\t\t{e}")
                            print(f"\t\tFail...")
                            if np.all(self.curr_weights['mv'] == 0):
                                self.curr_weights['mv'] = self.curr_weights['equal']
                            self.weights_records['mv'].append(
                                self.curr_weights['mv'])
                print(f'\t\tWeights: {self.curr_weights["mv"]}')

                # todo: u_mv_pbr1 shows unbounded
                print(f"\tRe-balance mv_pbr1")
                for i in range(self.max_tries):
                    try:
                        u_mv_pbr1 = oos_pbcv(pbr='mv_pbr1',
                                             X=X,
                                             mu=mu,
                                             Sigma=Sigma,
                                             beta=self.beta,
                                             alpha=self.alpha,
                                             gamma=self.gamma,
                                             div=self.div,
                                             bit=self.bit,
                                             R_target=self.R_target,
                                             k=self.k_fold)
                        # test
                        # u_mv_pbr1 = 1e-20
                        print(f'\t\tu_mv_pbr1: {u_mv_pbr1:.6e}')

                        self.u_records['mv_pbr1'].append(
                            (today_date, u_mv_pbr1))

                        w_mv_pbr1 = mv_pbr1_opt(
                            X, mu, Sigma, u_mv_pbr1, self.R_target)
                        w_mv_pbr1 = np.array(w_mv_pbr1)
                        self.curr_weights['mv_pbr1'] = w_mv_pbr1
                        self.weights_records['mv_pbr1'].append(w_mv_pbr1)
                        break
                    except Exception as e:
                        if i < self.max_tries - 1:
                            print(f"\t\t{e}")
                            print(f"\t\tRetrying... | {i+1}/{self.max_tries}")
                        else:
                            print(f"\t\t{e}")
                            print(f"\t\tFail...")
                            if np.all(self.curr_weights['mv_pbr1'] == 0):
                                self.curr_weights['mv_pbr1'] = self.curr_weights['equal']
                            self.weights_records['mv_pbr1'].append(
                                self.curr_weights['mv_pbr1'])
                weights_norm_change_mv = np.linalg.norm(
                    self.curr_weights['mv_pbr1'] - self.curr_weights['mv'])
                print(
                    f'\t\tWeights: {self.curr_weights["mv_pbr1"]}, norm change: {weights_norm_change_mv:.6e}')

                # cvar optimization:
                print(f"\tRe-balance cvar...")
                for i in range(self.max_tries):
                    try:
                        w_cv, alpha_cv = cvar_opt(X, self.beta, self.R_target)
                        w_cv = np.array(w_cv)
                        self.curr_weights['cvar'] = w_cv
                        self.weights_records['cvar'].append(w_cv)
                        break
                    except Exception as e:
                        if i < self.max_tries - 1:
                            print(f"\t\t{e}")
                            print(f"\t\tRetrying... | {i+1}/{self.max_tries}")
                        else:
                            print(f"\t\t{e}")
                            print(f"\t\tFail")
                            if np.all(self.curr_weights['cvar'] == 0):
                                self.curr_weights['cvar'] = self.curr_weights['equal']
                            self.weights_records['cvar'].append(
                                self.curr_weights['cvar'])
                print(f'\t\tWeights: {self.curr_weights["cvar"]}')

                print(f"\tRe-balance cvar_pbr...")
                for i in range(self.max_tries):
                    try:

                        u_cv_pbr = oos_pbcv(pbr='cvar_pbr',
                                            X=X,
                                            mu=mu,
                                            Sigma=Sigma,
                                            beta=self.beta,
                                            alpha=self.alpha,
                                            gamma=self.gamma,
                                            div=self.div,
                                            bit=self.bit,
                                            R_target=self.R_target,
                                            k=self.k_fold)
                        print(f'\t\tu_cv_pbr: {u_cv_pbr:.6e}')
                        self.u_records['cvar_pbr'].append(
                            (today_date, u_cv_pbr))

                        w_cv_pbr, alpha_cv_pbr = cvar_relax_opt(
                            X, self.beta, u_cv_pbr, None, self.R_target)  # note: we do not use U2 here
                        w_cv_pbr = np.array(w_cv_pbr)
                        self.curr_weights['cvar_pbr'] = w_cv_pbr
                        self.weights_records['cvar_pbr'].append(
                            w_cv_pbr)
                        break
                    except Exception as e:
                        if i < self.max_tries - 1:
                            print(f"\t\t{e}")
                            print(f"\t\tRetrying... | {i+1}/{self.max_tries}")
                        else:
                            print(f"\t\t{e}")
                            print(f"\t\tFail")
                            if np.all(self.curr_weights['cvar_pbr'] == 0):
                                self.curr_weights['cvar_pbr'] = self.curr_weights['equal']
                            self.weights_records['cvar_pbr'].append(
                                self.curr_weights['cvar_pbr'])
                weights_norm_change_cvar = np.linalg.norm(
                    self.curr_weights['cvar_pbr'] - self.curr_weights['cvar'])
                print(
                    f'\t\tWeights: {self.curr_weights["cvar_pbr"]}, norm change: {weights_norm_change_cvar:.6e}')
            adjust_count += 1


# %%
# initialize
figure_dir = '/Users/ellenzh/Library/CloudStorage/Dropbox/optim_project/fiigures_ellen'
start_date = datetime(2010, 1, 3)
end_date = datetime(2020, 1, 3)

df_list = [ff5_df, ff10_df]
lookback_list = [60, 120]
rebalance_freq_list = [63, 21, 11]
R_target_list = [None, 0, 1e-3, 2e-3]

param_list = list(product(df_list, lookback_list,
                  rebalance_freq_list, R_target_list))
for df, lookback, rebalance_freq, R_target in param_list:
    print(f'\n{"%"*80}')
    print(
        f'lookback: {lookback}, rebalance_freq: {rebalance_freq}, R_target: {R_target}')
    print(f'{"%"*80}')
    # lookback = 50
    # rebalance_freq = 21
    R_target_str = 'None' if R_target is None else f'{R_target:.3f}'
    U_tuning_params = {
        'alpha': 0.4,
        'beta': 0.95,
        'div': 5,
        'bit': 0.05,
        'gamma': 0.9,
        'k': 5
    }

    backtester = Backtester(data=df,
                            lookback=lookback,
                            rebalance_freq=rebalance_freq,
                            R_target=R_target,
                            start_date=start_date,
                            end_date=end_date,
                            U_tuning_params=U_tuning_params)

    # %%
    backtester.run()

    # %%
    # Final NAV
    print("\nFinal NAV:")
    for stra, nav in backtester.navs.items():
        print(f"{stra}: {nav[-1]:.4f}")

    # %%
    # Sharpe Ratio
    sharpe_dict = {}
    print("\nSharpe Ratio:")
    for stra, nav in backtester.navs.items():
        daily_ret = np.array(nav[1:]) / np.array(nav[:-1]) - 1
        mu = np.mean(daily_ret)
        sigma = np.std(daily_ret, ddof=1)
        sr = mu / sigma * np.sqrt(252)
        print(f"{stra}: {sr:.4f}")
        sharpe_dict[stra] = sr

    # %%
    # NAV Curve
    dates = backtester.backtest_data['Date'].iloc[lookback-2:].to_numpy()
    plt.figure(figsize=(12, 6))
    for stra, nav in backtester.navs.items():
        plt.plot(dates, nav, linewidth=1.5,
                 label=f'{stra} (Sharpe: {sharpe_dict[stra]:.3f})')
    plt.title(
        f'NAV cross Time from {start_date.strftime(format="%Y-%m-%d")} to {end_date.strftime(format="%Y-%m-%d")} | lookback = {lookback} | frequency = {rebalance_freq} | R_target = {R_target_str}')
    plt.xlabel('Date')
    plt.ylabel('NAV')
    plt.legend(loc='upper left', fontsize=8)
    plt.grid()
    n_industries = len(backtester.backtest_data.columns) - 1
    fig_path = f'NAV_FF{n_industries}_{start_date.strftime(format="%Y%m%d")}_{end_date.strftime(format="%Y%m%d")}_{lookback}_{rebalance_freq}_{R_target_str}.pdf'
    fig_path = figure_dir + '\\' + fig_path
    plt.savefig(fig_path, bbox_inches='tight')
    # print(f"Figure saved to {fig_path}")
