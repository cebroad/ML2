# Claire Broad
# MSAN 630: Advanced Machine Learning
# Professor Yannet Interian
# Assignment 1: Collaborative Filtering
# Due February 2, 2017

import argparse
import re
import os
import csv
import math
import collections as coll
import numpy as np
import time

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def parse_file(filename):
    """
    Given a filename outputs user_ratings and movie_ratings dictionaries

    Input: filename

    Output: user_ratings, movie_ratings
        where:
            user_ratings[user_id] = {movie_id: rating}
            movie_ratings[movie_id] = {user_id: rating}
    """
    user_ratings = {}
    movie_ratings = {}
    with open(filename, 'r') as file:
        filereader = csv.reader(file)
        for line in filereader:
            movie = int(line[0])
            user = int(line[1])
            rating = float(line[2])
            user_ratings.setdefault(user, {})
            user_ratings[user][movie] = rating
            movie_ratings.setdefault(movie, {})
            movie_ratings[movie][user] = rating
    return user_ratings, movie_ratings


def compute_average_user_ratings(user_ratings):
    """ Given a the user_rating dict compute average user ratings

    Input: user_ratings (dictionary of user, movies, ratings)
    Output: ave_ratings (dictionary of user and ave_ratings)
    """
    ave_ratings = {}
    for user in user_ratings:
        ratings = user_ratings[user]
        allratings = [ratings[movie] for movie in ratings]
        ave_ratings[user] = np.mean(allratings)
    return ave_ratings


def compute_user_similarity(d1, d2, ave_rat1, ave_rat2):
    """ Computes similarity between two users

        Input: d1, d2, (dictionary of user ratings per user) 
            ave_rat1, ave_rat2 average rating per user (float)
        Ouput: user similarity (float)
    """
    common_movies = set(d1) & set(d2)
    if len(common_movies) > 0:
        u1_vec = [d1[movie] - ave_rat1 for movie in common_movies]
        u2_vec = [d2[movie] - ave_rat2 for movie in common_movies]
        num = np.dot(u1_vec, u2_vec)
        denom = math.sqrt(np.dot(u1_vec, u1_vec)*np.dot(u2_vec, u2_vec))
        if denom == 0.0:
            return 0.0
        return num/denom
    return 0.0


def main():
    """
    This function is called from the command line via
    
    python cf.py --train [path to filename] --test [path to filename]
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    user_ratings_train, movie_ratings_train = parse_file(train_file)
    user_ratings_test, movie_ratings_test = parse_file(test_file)
    user_avg_train = compute_average_user_ratings(user_ratings_train)
    sq_errors = []
    abs_errors = []
    final_data = []
    for movie in movie_ratings_test:                      # Movie A
        # Users who rated movie A
        raters = movie_ratings_train[movie]

        # Users who are in the test set for movie A
        users = movie_ratings_test[movie]

        # Training set ratings of movie A {User: Rating}
        movie_ratings = movie_ratings_train[movie]

        # Similarities b/n raters + users {User:{Rater:Sim,...},...}
        user_similarities = {user:{rater:compute_user_similarity(user_ratings_train[user],user_ratings_train[rater],user_avg_train[user],user_avg_train[rater]) for rater in raters} for user in users}

        # Numerator sums
        rater_diffs = {user:sum([user_similarities[user][rater]*(movie_ratings[rater] - user_avg_train[rater]) for rater in raters]) for user in users}

        # Scaling factors/denominators
        denom = {user:sum([abs(sim) for sim in user_similarities[user].values()]) for user in users}

        for user in users:
            if denom[user] == 0.0:
                pred = user_avg_train[user]
            else:
                pred = user_avg_train[user] + rater_diffs[user]/denom[user]
            sq_errors.append((pred - user_ratings_test[user][movie]) ** 2)
            abs_errors.append(abs(pred - user_ratings_test[user][movie]))
            final_data.append([movie, user, user_ratings_test[user][movie], pred])

    rmse = math.sqrt(np.mean(sq_errors))
    print rmse
    mae = np.mean(abs_errors)
    print mae
    with open('predictions.txt', 'w') as predictions:
        pred_writer = csv.writer(predictions)
        for row in final_data:
            pred_writer.writerow(row)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    # print "Total time elapsed: " + str(end - start) + " seconds"
