"""
The get_data function should load the raw data in any format from any place and return it
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Text, Tuple

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from utils import load_data, high_corr, in_out_array, plot, statistics
from models import train, predict

logger = structlog.getLogger(__name__)


def parse_arguments(arguments_list: List) -> Namespace:
    """Parse arguments list"""
    parser = ArgumentParser(__name__)
    parser.add_argument('--data_path', type=Path, help='data path')
    parser.add_argument('--model_path', type=Path, help='a path to save and load model from')
    parser.add_argument('--data_slice', type=str, help='slice of the data you want to do predictions on.',
                        choices=['General', 'Summer', 'Fall', 'Winter', 'Spring', 'Weekend', 'COVID Weekend', 'COVID'])
    parser.add_argument('--phase', type=str, help='the process you wanna do on your selected part of data',
                        choices=['train', 'predict', 'statistics'])
    parser.add_argument('--config_path', type=Path, help='Path to data acquisition config file')
    parser.add_argument('--th', type=int, help='threshold for correlation filtering', default='0.4')

    return parser.parse_args(arguments_list)


def retrieve_data(arguments: Namespace) -> Tuple:

    init_df, init_stats_df, init_corr_df = load_data(Path(arguments.data_path))

    if arguments.data_slice == 'Summer':
        df = init_df[(init_df.June == 1) | (init_df.July == 1) | (init_df.August == 1)]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, y_train = x[:-92], y[:-92]
        x_test, y_test = x[-92:], y[-92:]

    if arguments.data_slice == 'Fall':
        df = init_df[(init_df.September == 1) | (init_df.October == 1) | (init_df.November == 1)]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, y_train = x[:-92], y[:-92]
        x_test, y_test = x[-92:], y[-92:]

    if arguments.data_slice == 'Winter':
        df = init_df[(init_df.December == 1) | (init_df.January == 1) | (init_df.March == 1)]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, y_train = x[:-92], y[:-92]
        x_test, y_test = x[-92:], y[-92:]

    if arguments.data_slice == 'Spring':
        df = init_df[(init_df.March == 1) | (init_df.April == 1) | (init_df.May == 1)]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, y_train = x[:-92], y[:-92]
        x_test, y_test = x[-92:], y[-92:]
    if arguments.data_slice == 'Weekend':
        df = init_df[(init_df['Saturday'] == 1) | (init_df['Sunday'] == 1) | (init_df['Sun_Hol'] == 1)]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    if arguments.data_slice == 'COVID Weekend':
        df = init_df[((init_df['Saturday'] == 1) & (init_df['COVID19'] >= 0.01)) |
                              ((init_df['Sunday'] == 1) & (init_df['COVID19'] >= 0.01)) |
                              ((init_df['Sun_Hol'] == 1) & (init_df['COVID19'] >= 0.01))]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    if arguments.data_slice == 'COVID':
        df = init_df[init_df['COVID19'] >= 0.01]
        filtered_df = high_corr(df, arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    else:

        filtered_df = high_corr(init_df[:2434], arguments.th)
        x, y = in_out_array(filtered_df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

    return x_train, x_test, y_train, y_test


def run(arguments_list: List = None):
    """Run step"""
    arguments = parse_arguments(arguments_list=arguments_list)
    x_train, x_test, y_train, y_test = retrieve_data(arguments)


if __name__ == '__main__':
    run()
