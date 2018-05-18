import csv
import os.path
import numpy as np
import pandas as pd
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


def plot_attack_defense(df):
    sns.lmplot(x='Attack', y='Defense', data=df, fit_reg=False, hue='Type1', palette=pkmn_type_colors)


def generation(row):
    if row.Number < 152:
        return 1
    else:
        return 2


def factor_plot(gf):
    sns.factorplot(
        x='Attack',
        y='Defense',
        data=gf,
        hue='Type1',
        palette=pkmn_type_colors,
        col='Gen',
        kind='swarm',
        sharex=False,
    )
    plt.show()


def facet_grid(gf):
    g = sns.FacetGrid(
        gf,
        col='Gen',
        hue='Type1',
        palette=pkmn_type_colors,
    )
    g = g.map(plt.scatter, 'Attack', 'Defense')
    plt.show()


def nmin_indexes(arr, n):
    mins = {}
    for i in range(n):
        min_index = np.argmin(arr)
        mins[min_index] = arr[min_index]
        arr[min_index] = np.max(arr)

    for i in mins.keys():
        arr[i] = mins[i]

    return mins


def setup_prediction(o, r, k):
    def predict(x):
        d = np.sqrt(np.sum(np.square(np.subtract(o, x)), axis=1))
        nn = nmin_indexes(d, k)
        nn_values = []
        for i in nn.keys():
            nn_values.append(r[i])
        unique, counts = np.unique(nn_values, return_counts=True)
        count_map = dict(zip(counts, unique))
        return count_map[np.max(counts)]
    return predict


def run():
    df = None
    if os.path.isfile('johto.csv'):
        print("Loading from file")
        df = pd.read_csv('johto.csv')
    else:
        print("No local file")
        return

    df['Gen'] = df.apply(lambda row: generation(row), axis=1)
    gf = df[df.Type1.isin(['Grass', 'Fire'])]

    gen1 = gf[gf.Gen == 1]  # 'Train' data
    gen2 = gf[gf.Gen == 2]  # Test data

    observations = gen1.as_matrix(['Attack', 'Defense'])
    results = gen1.as_matrix(['Type1'])

    predict = setup_prediction(observations, results, 7)

    correct = 0
    total = 0

    for row in gen2.itertuples():
        total += 1
        prediction = predict(np.array([int(row.Attack), int(row.Defense)]))
        if prediction == row.Type1:
            correct += 1
        correctness = int(correct/total * 100)
        print("kNN predicted: {0}, answer was: {1}. Correctness: {2}".format(prediction, row.Type1, correctness))

    all_points = np.dstack(np.meshgrid(np.arange(25, 130),np.arange(25, 130))).reshape(-1,2)
    borders = []

    for i in all_points:
        borders.append(predict(i))

    everything = pd.DataFrame(all_points, columns=['Attack', 'Defense'])
    everything['Type1'] = borders
    plot_attack_defense(everything)

    facet_grid(gf)


run()
