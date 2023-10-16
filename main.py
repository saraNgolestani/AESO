"""
A minni application for training and evaluating ML models on AESO energy consumption data
"""
import os.path
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple, Dict
import structlog
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from utils import load_data, high_corr, in_out_array, plot, statistics, save_model, load_model, load_yaml_config
from models import train, predict

logger = structlog.getLogger(__name__)


def parse_arguments(config: Dict) -> Namespace:
    """Parse arguments list"""
    parser = ArgumentParser(__name__)
    parser.add_argument('--data_path', type=Path, help='data path',
                        default=config['PATHS']['Data'])
    parser.add_argument('--model_path', type=Path, help='a path to save and load model from')
    parser.add_argument('--data_slice', type=str, help='part of data you want to do prediction on',
                        default=config['DATA_SLICE'],
                        choices=['General', 'Summer', 'Fall', 'Winter', 'Spring', 'Weekend', 'COVID Weekend', 'COVID'])
    parser.add_argument('--phase', type=str, help='the process you wanna do on your selected part of data',
                        choices=['train', 'predict', 'statistics'], default=config['PHASE'])
    parser.add_argument('--th', type=int, help='threshold for correlation filtering', default=config['TH'])
    args = parser.parse_args()
    return args


def retrieve_data(arguments: Namespace) -> Tuple:
    """
    Retrieve and process data based on the selected data slice.

    Parameters:
    - arguments (Namespace): Parsed command line arguments and configuration settings.

    Returns:
    - x_train, x_test, y_train, y_test, filtered_df (tuple): Processed data and DataFrame.
    """

    init_df, init_stats_df, init_corr_df = load_data(Path(arguments.data_path))

    if arguments.data_slice == 'Summer':
        df = init_df[(init_df.June == 1) | (init_df.July == 1) | (init_df.August == 1)]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, y_train = x[:-92], y[:-92]
        x_test, y_test = x[-92:], y[-92:]

    elif arguments.data_slice == 'Fall':
        df = init_df[(init_df.September == 1) | (init_df.October == 1) | (init_df.November == 1)]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, y_train = x[:-92], y[:-92]
        x_test, y_test = x[-92:], y[-92:]

    elif arguments.data_slice == 'Winter':
        df = init_df[(init_df.December == 1) | (init_df.January == 1) | (init_df.March == 1)]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, y_train = x[:-92], y[:-92]
        x_test, y_test = x[-92:], y[-92:]

    elif arguments.data_slice == 'Spring':
        df = init_df[(init_df.March == 1) | (init_df.April == 1) | (init_df.May == 1)]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, y_train = x[:-92], y[:-92]
        x_test, y_test = x[-92:], y[-92:]
    elif arguments.data_slice == 'Weekend':
        df = init_df[(init_df['Saturday'] == 1) | (init_df['Sunday'] == 1) | (init_df['Sun_Hol'] == 1)]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    elif arguments.data_slice == 'COVID Weekend':
        df = init_df[((init_df['Saturday'] == 1) & (init_df['COVID19'] >= 0.01)) |
                              ((init_df['Sunday'] == 1) & (init_df['COVID19'] >= 0.01)) |
                              ((init_df['Sun_Hol'] == 1) & (init_df['COVID19'] >= 0.01))]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    elif arguments.data_slice == 'COVID':
        df = init_df[init_df['COVID19'] >= 0.01]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    else:
        filtered_df = high_corr(init_df[:2434], arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    return x_train, x_test, y_train, y_test, filtered_df


def run(config: Dict):
    """
        Main execution function for training, predicting, or computing statistics.

        Parameters:
        - config (dict): Configuration settings loaded from a YAML file.
    """

    arguments = parse_arguments(config)
    x_train, x_test, y_train, y_test, filtered_df = retrieve_data(arguments)

    if arguments.phase == 'train':
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5, 0.2],
            'max_depth': [3, 4, 5, 6]
        }

        # Create the Gradient Boosting Regressor model
        init_model = GradientBoostingRegressor(random_state=42)
        model = train(param_grid, init_model, x_train, y_train)
        y_pred = predict(model, x_test, y_test)
        plot(y_pred, y_test, (16, 12), 'Data Points', f'Energy Values',
             f'Actual Vs Predicted Values GradientBoostRegressor on {arguments.data_slice} Data')
        save_model(model, os.path.join('./models', arguments.data_slice))

    if arguments.phase == 'predict':
        model = load_model(os.path.join('./models', arguments.data_slice))
        y_pred = predict(model, x_test, y_test)
        plot(y_pred, y_test, (16, 12), 'Data Points', f'Energy Values',
             f'Actual Vs Predicted Values GradientBoostRegressor on {arguments.data_slice} Data')

    if arguments.phase == 'statistics':
        statistics(filtered_df)


if __name__ == '__main__':
    config = load_yaml_config(Path('config.yaml'))
    run(config)
