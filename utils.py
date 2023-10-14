from pathlib import Path
from typing import List, Text, Tuple

from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from numpy import array
import joblib


def load_data(data_path: Path) -> Tuple:
    init_df = pd.read_excel(data_path, sheet_name='Dataset')
    init_stats_df = pd.read_excel(data_path, sheet_name='Statistics')
    init_corr_df = pd.read_excel(data_path, sheet_name='Correlation')
    return init_df, init_stats_df, init_corr_df


def high_corr(df: pd.DataFrame, th: float) -> pd.DataFrame:
    """
    Commonly, correlations are categorized as follows:
    0.00 to 0.19: Very weak or no correlation
    0.20 to 0.34: Weak correlation
    0.35 to 0.59: Moderate correlation
    0.60 to 0.79: Strong correlation
    0.80 to 1.00: Very strong correlation
    """
    corr = df.corr()
    corr = pd.DataFrame(corr)
    corr_high = corr[abs(corr.Energy) >= th]
    new_df = df[corr_high.index]
    for col in new_df.columns:
        if new_df[f'{col}'].isnull().sum() > 0:
            new_df[f'{col}'].fillna((new_df[f'{col}'].mean()), inplace=True)

    return new_df


def in_out_array(df: pd.DataFrame) -> Tuple[array, array]:

    return np.array(df[df.columns[1:]]), np.array(df[df.columns[0]])


def save_model(model: GradientBoostingRegressor, model_name: str):
    joblib.dump(model, model_name.join('.pkl'))
    return


def load_model(model_name: str) -> GradientBoostingRegressor:
    model = joblib.load(model_name.join('.pkl'))
    return model


def plot(y_pred, y_test, figsize: (int, int), xlabel: str, ylabel: str, title: str):
    plt.figure(figsize=figsize)
    x = np.arange(len(y_test))  # Create an array of indices
    plt.plot(x, y_test, label='Actual', color='blue', marker='o', linestyle='-', markersize=5)
    plt.plot(x, y_pred, label='Predicted', color='red', marker='x', linestyle='--', markersize=5)
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    plt.title(f'{title}')
    plt.legend()
    plt.show()
    return


def statistics():
    return

