from pathlib import Path
from typing import List, Text, Tuple

import numpy as np
import pandas as pd
from numpy import array


def load_data(data_path : Path) -> Tuple:
    init_df = pd.read_excel(data_path, sheet_name='Dataset')
    init_stats_df = pd.read_excel(data_path, sheet_name='Statistics')
    init_corr_df = pd.read_excel(data_path, sheet_name='Correlation')
    return init_df, init_stats_df, init_corr_df


def high_corr(df: pd.DataFrame, th: float) -> pd.DataFrame:
    corr = df.corr()
    corr = pd.DataFrame(corr)
    corr_high = corr[abs(corr.Energy) >= th]
    new_df = df[corr_high.index]
    for col in new_df.columns:
        if new_df[f'{col}'].isnull():
            new_df[f'{col}'].fillna((new_df[f'{col}'].mean()), inplace=True)

    return new_df


def in_out_array(df: pd.DataFrame) -> Tuple[array, array]:

    return np.array(df[df.columns[1:]]), np.array(df[df.columns[0]])


def save_model():
    return


def load_model():
    return


def plot():
    return


def statistics():
    return

