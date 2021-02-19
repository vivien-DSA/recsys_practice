###############################################LOAD THE libraries ########################################################

import time

import pandas
import numpy as np
from pathlib import Path
from collections import defaultdict


from  surprise.prediction_algorithms.algo_base import AlgoBase


################################################## RETRIEVE TOP MOVIES AND RECOMMENDATIONS################################################

def create_user_movieId_trueRatings_dataframe(data):
    return data[['userId', 'movieId', 'rating']].to_numpy()

def get_top_n(predictions, n=10) -> defaultdict:
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def create_top_n_dataframe(top_n, data):
    # Save the top_n in a data frame    
    dataf = pandas.DataFrame()
    # To retrieve the true rating we will check on the test result
    test_a = np.array(data)
    for uid, user_ratings in top_n.items():
        for m, r in user_ratings:
            tr = test_a[(test_a[:,0] == uid) & (test_a[:,1] == m)].item(2)
            dataf = pandas.concat([dataf, pandas.DataFrame([[int(uid), int(m), round(r, 2), tr]],
                                               columns = ['UserId', 'movieId', 'PredictedRating', 'TrueRating'])],
                          ignore_index=True)
    return dataf

def create_recommendation_dataframe(top_n_df, movies, userId):
    movies_recom_all_users = top_n_df.merge(movies, how='left', on=['movieId'])
    output = movies_recom_all_users[movies_recom_all_users['UserId'] == userId].drop(columns = ['UserId', 'movieId'])
    cols = ['title', 'genres', 'PredictedRating', 'TrueRating']
    output = output[cols]
    return output

def get_user_recommendation(model: AlgoBase, user_id: int, k: int, data: pandas.DataFrame, movies : pandas.DataFrame
                           ) -> pandas.DataFrame:
    """Makes movie recommendations a user.
    
    Parameters
    ----------
        model : AlgoBase
            A trained surprise model
        user_id : int
            The user for whom the recommendation will be done.
        k : int
            The number of items to recommend.
        data :  pandas.DataFrame
            The data needed to do the recommendation ( ratings dataframe ).
        movies : pandas.DataFrame
            The dataframe containing the movies metadata (title, genre, etc)
        
    Returns
    -------
    pandas.Dataframe
        A dataframe with the k movies that will be recommended the user. The dataframe should have the following
        columns (movie_name : str, movie_genre : str, predicted_rating : float, true_rating : float)
        
    Notes
    -----
    - You should create other functions that are used in this one and not put all the code in the same function.
        For example to create the final dataframe, instead of implemented all the code
        in this function (get_user_recommendation), you can create a new one (create_recommendation_dataframe)
        that will be called in this function.
    - You can add other arguments to the function if you need to.
    """
    data_to_test = create_user_movieId_trueRatings_dataframe(data)
    predictions = model.test(data_to_test)
    top_n = get_top_n(predictions, n=k)
    top_n_df = create_top_n_dataframe(top_n, data_to_test)
    recommendation_df = create_recommendation_dataframe(top_n_df, movies, user_id).head(k)  
    return recommendation_df


#####################################################################################################################