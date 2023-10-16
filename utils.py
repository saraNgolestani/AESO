from pathlib import Path
from typing import Tuple, Dict
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from numpy import array
import joblib
from statsmodels.stats.stattools import jarque_bera
from yaml import dump, safe_load


def load_data(data_path: Path) -> Tuple:

    """
    Load data from an Excel file into pandas dataframes.
    :param data_path: The path to the Excel file.
    :return: Tuple containing three dataframes: (init_df, init_stats_df, init_corr_df)
    """"""
    """

    init_df = pd.read_excel(data_path, sheet_name='Dataset')
    init_stats_df = pd.read_excel(data_path, sheet_name='Statistics')
    init_corr_df = pd.read_excel(data_path, sheet_name='Correlation')
    return init_df[:2434], init_stats_df, init_corr_df


def high_corr(df: pd.DataFrame, th: float) -> pd.DataFrame:
    """
    Filter a dataframe to retain columns with high correlation to the 'Energy' column.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - th (float): The correlation threshold.

    Returns:
    A new dataframe containing only high-correlation columns.
    
    """"""
    Commonly, correlations are categorized as follows:
    0.00 to 0.19: Very weak or no correlation
    0.20 to 0.39: Weak correlation
    0.40 to 0.59: Moderate correlation
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
    """
        Extract features and target variable from a dataframe and return them as numpy arrays.

        Parameters:
        - df (pd.DataFrame): The input dataframe.

        Returns:
        A tuple containing two arrays: (features, target).
        """
    return np.array(df[df.columns[1:]]), np.array(df[df.columns[0]])


def save_model(model: GradientBoostingRegressor, model_name: str):
    """
        Save a scikit-learn model to a file using joblib.

        Parameters:
        - model (GradientBoostingRegressor): The trained model to be saved.
        - model_name (str): The name of the saved model file.

        Returns:
        A message indicating that the model has been saved.
        """
    joblib.dump(model, model_name+'.pkl')
    return 'model saved'


def load_model(model_name: str) -> GradientBoostingRegressor:
    """
    Load a scikit-learn model from a file using joblib.

    Parameters:
    - model_name (str): The name of the saved model file.

    Returns:
    The loaded scikit-learn model.
    """
    model = joblib.load(model_name+'.pkl')
    return model


def plot(y_pred, y_test, fig_size: (int, int), x_label: str, y_label: str, title: str):
    """
        Create a line plot to visualize actual and predicted values.

        Parameters:
        - y_pred: Predicted values.
        - y_test: Actual values.
        - fig_size (tuple): Figure size (width, height).
        - x_label (str): Label for the x-axis.
        - y_label (str): Label for the y-axis.
        - title (str): Title of the plot.
    """
    plt.figure(figsize=fig_size)
    x = np.arange(len(y_test))  # Create an array of indices
    plt.plot(x, y_test, label='Actual', color='blue', marker='o', linestyle='-', markersize=5)
    plt.plot(x, y_pred, label='Predicted', color='red', marker='x', linestyle='--', markersize=5)
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'{y_label}')
    plt.title(f'{title}')
    plt.legend()
    plt.show()
    return


def statistics(df: pd.DataFrame):
    """
        Calculate and print various statistics for columns in a dataframe.

        Parameters:
        - df (pd.DataFrame): The input dataframe.
    """
    corr_df = df.corr()
    print(corr_df.Energy)
    for col in df.columns:
        df_col = df[f'{col}']
        print(f'For {col} we have \n mean of : {df_col.mean()} \n var of: {df_col.var()} \n '
              f'min of {df_col.min()} \n max of {df_col.max()} \n skewness of {df_col.skew()} \n '
              f'kurtosis of {df_col.kurtosis()} \n and the goodness of fittest of {jarque_bera(np.array(df_col))[0]}')

    return


def load_yaml_config(config_path: Path) -> Dict:
    """Load YAML config file and return a parsed, ready to consume, dictionary

    param config_path: location of the config file
    :return: config dictionary
    """
    with config_path.open(mode='r') as stream:
        config = safe_load(stream)
    return config


def write_yaml_config(config: Dict, path: Path):
    """Write config object (dictionary) to a YAML config file

    param path:
    :param config:
    :config: config object
    :param config_path: location to write the config file
    """
    with path.open(mode='w') as stream:
        dump(config, stream)