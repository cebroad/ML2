import numpy as np
import argparse
from sklearn.tree import DecisionTreeClassifier
from math import log
import csv

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    parser.add_argument('--numTrees', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def adaboost(X, y, num_iter):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is in {-1, 1}^n
    """
    trees = []
    trees_weights = []
    obs_weights = [float(1)/len(y)] * len(y)
    for i in range(num_iter):
        tree = DecisionTreeClassifier(max_depth = 1)
        tree.fit(X, y, sample_weight = obs_weights)
        trees.append(tree)
        pred = tree.predict(X)
        err_num = 0
        err_denom = sum(obs_weights)
        eval = []
        for j in range(len(pred)):
            if pred[j] != y[j]:
                err_num += obs_weights[j]
                eval.append(1)
            else:
                eval.append(0)
        err = err_num/err_denom
        if err > 0:
            alpha = log((1-err)/err)
        else:
            alpha = np.inf
        trees_weights.append(alpha)
        # print obs_weights, err, alpha, pred
        for k in range(len(obs_weights)):
            obs_weights[k] = obs_weights[k]*np.exp(alpha*eval[k])
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y

    assume Y in {-1, 1}^n
    """
    # your code here

    obs_num, feat_num = X.shape

    pred_array = [[] for _ in range(obs_num)]

    for i in range(len(trees)):
        pred = trees[i].predict(X)
        for j in range(obs_num):
            pred_array[j].append(pred[j] * trees_weights[i])

    Yhat = []

    for i in range(len(pred_array)):
        Ysum = sum(pred_array[i])
        if Ysum >= 0:
            Yhat.append(1.)
        else:
            Yhat.append(-1.)
    return Yhat


def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row.
    """
    # your code here
    X = []
    Y = []
    with open(filename, 'r') as file:
        for line in file:
            data = line.rstrip().split(',')
            X.append(data[:-1])
            Y.append(float(data[-1]))
    X = np.array(X).astype("float")
    return X, Y

def new_label(Y):
    """ Transforms a vector od 0s and 1s in -1s and 1s.
    """
    return [-1. if y == 0. else 1. for y in Y]

def old_label(Y):
    return [0. if y == -1. else 1. for y in Y]

def accuracy(y, pred):
    acc_num = 0
    for i in range(len(y)):
        if y[i] == pred[i]:
            acc_num += 1
    return acc_num / float(len(y))

def main():
    """
    This code is called from the command line via
    
    python adaboost.py --train [path to filename] --test [path to filename] --numTrees 
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    num_trees = int(args['numTrees'][0])
    print train_file, test_file, num_trees

    # your code here

    X_train, Y_train = parse_spambase_data(train_file)
    X_test, Y_test = parse_spambase_data(test_file)

    trained_trees, trained_weights = adaboost(X_train, new_label(Y_train), num_trees)

    # print trained_weights

    Yhat_train = adaboost_predict(X_train, trained_trees, trained_weights)

    Yhat_test = adaboost_predict(X_test, trained_trees, trained_weights)

    ## here print accuracy and write predictions to a file
    acc = accuracy(Y_train, old_label(Yhat_train))
    acc_test = accuracy(Y_test, old_label(Yhat_test))

    print("Train Accuracy %.4f" % acc)
    print("Test Accuracy %.4f" % acc_test)

    X_test_output = X_test.tolist()

    assemble = []

    for i in range(len(Yhat_test)):
        assemble.append(X_test_output[i] + [Y_test[i]] + [Yhat_test[i]])

    with open('predictions.txt', 'w') as predfile:
        predwriter = csv.writer(predfile)
        for line in assemble:
            predwriter.writerow(line)


if __name__ == '__main__':
    main()

