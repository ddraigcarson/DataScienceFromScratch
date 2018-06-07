from time import time
from random import seed, sample
from random import randrange
from csv import reader
from math import sqrt
from os import path
import numpy as np
import pandas as pd


def create_converters():
    i1 = lambda i: int(i[-1])
    i2 = lambda i: int(i[2:])
    i1_cols = ("1", "3", "6", "7", "9", "10", "12", "14", "15", "17", "19", "20")
    i2_cols = ("4")

    converters = dict()
    for col in i1_cols:
        converters[col] = i1
    for col in i2_cols:
        converters[col] = i2

    return converters


def load_csv(filename):
    if path.isfile(filename):
        print("Loading from file")
        headers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                   "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
        dataset = pd.read_csv('data-code.csv', header=None, names=headers, index_col=False, converters=create_converters())
        return dataset
    print("No data")
    return None


# Used for cross validation splits and creating samples for the random trees
def get_samples_from_dataset(dataset, n_samples, sample_size=0):
    dataset_split = list()
    if sample_size == 0:
        sample_size = int(len(dataset)/n_samples)
    for i in range(n_samples):
        dataset_split.append(dataset.sample(sample_size))
    return dataset_split


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = get_samples_from_dataset(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        print("CLASS {0} ".format(class_value))
        for group in groups:
            print("GROUP {0} ".format(group))
            size = len(group)
            print("size {0} ".format(size))
            if size == 0:
                continue
            print("ROW")
            print([row[-1] for row in group])
            count = [row[-1] for row in group].count(class_value)
            print("count of class_value {0}, is {1}".format(class_value, count))
            proportion = count / float(size)
            print("proportion {0}".format(proportion))
            #  sum all (p * 1-p) values, this is gini index
            gini += (proportion * (1.0 - proportion))
    print("gini {0}".format(gini))
    return gini

def gini_impurity(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        print("CLASS {0} ".format(class_value))
        for group in groups:
            print("GROUP {0} ".format(group))
            if group.empty:
                continue
            unique, counts = np.unique(group[group.columns[-1]], return_counts=True)
            all_counts = dict(zip(unique, counts))
            count = all_counts[class_value]
            print("count {0}".format(count))
            size = len(group)
            proportion = count / float(size)
            print("proportion {0}".format(proportion))
            gini += (proportion * (1.0 - proportion))
    return gini


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


def random_forest(train, test, max_depth, min_size, n_trees, n_features):
    trees = list()
    samples = get_samples_from_dataset(train, n_trees, len(train))
    for sample in samples:
        trees.append(build_tree(sample, max_depth, min_size, n_features))
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)


def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features) # Choose the root node
    split(root, max_depth, min_size, n_features, 1)
    return root


def get_split(dataset, n_features):
    b_index, b_value, b_score, b_groups = 999, 999, 999, None

    class_values = list(set(row[-1] for row in dataset)) # unique list of the outputs
    features = np.random.choice(len(dataset.columns)-1, n_features, False)

    for index in features:
        for row in dataset.values:
            groups = test_split(dataset.columns[index], row[index], dataset)
            gini = gini_index(groups, class_values)
            #if gini < b_score:
            #    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    #return {'index': b_index, 'value': b_value, 'groups': b_groups}


def test_split(column, value, dataset):
    left = dataset[dataset[column] < value]
    right = dataset[dataset[column] >= value]
    return left, right


seed(1)

start = time()
dataset = load_csv('data.csv')
end = time()
print("Read file in {0}".format((end-start)))

n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1


