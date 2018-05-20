import csv
import os.path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                    ]


def plot(df):
    sns.countplot(x='Type1', data=df)
    sns.lmplot(x='Attack', y='Defense', data=df, fit_reg=True)
    plt.show()


def get_data_from_file(file):
    df = None
    if os.path.isfile(file):
        print("Loading from file")
        df = pd.read_csv(file)
    else:
        print("No local file")
        return
    return df


def dedc(m, c, x, y):
    return (2/len(x)) * (m * x + c - y)


def dedm(m, c, x, y):
    return ((2 * x)/len(x)) * (m * x + c - y)


def sum_of_squared_error(m, c, x, y):
    return np.sum(error(m, c, x, y)**2)/len(x)


def error(m, c, x, y):
    return y - predict(m, c, x)


def predict(m, c, x):
    return m * x + c


def get_by_types(df, types):
    return df[df.Type1.isin(types)]


def linear_regression(df, x_column, y_column):
    learning_step = 0.0001
    m = 0
    c = 0
    n = 1000
    print(df[y_column])
    y = df[x_column].values     # numpy array
    x = df[y_column].values    # numpy array
    for i in range(n):
        m = m - learning_step * np.sum(dedm(m, c, x, y))
        c = c - learning_step * np.sum(dedc(m, c, x, y))
        e = sum_of_squared_error(m, c, x, y)
        print('e: {0}, m: {1}, c: {2} '.format(e, m, c))


def run():
    # df = get_data_from_file('johto.csv')
    # linear_regression(df, 'Attack', 'Defense')
    df = get_data_from_file('data.csv')
    linear_regression(df, 'x', 'y')
    #plot(df)


run()
