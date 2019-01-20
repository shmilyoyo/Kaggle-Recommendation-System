# DATA MUNGING

import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from pathlib import Path


def mungingData(inputDataRootPath, outputDataRootPath):
    inputDataRootPath = Path(inputDataRootPath)
    outputDataRootPath = Path(outputDataRootPath)

    # load data
    interactions_df = pd.read_csv(
        str(inputDataRootPath / "users_interactions.csv"))

    """
    weight on interactions
    """
    # weight on interactions
    event_type_strength = {
        'VIEW': 1.0,
        'LIKE': 2.0,
        'BOOKMARK': 2.5,
        'FOLLOW': 3.0,
        'COMMENT CREATED': 4.0,
    }

    interactions_df["eventStrength"] = interactions_df["eventType"].apply(
        lambda x: event_type_strength[x])

    """
    filter the data on # of interactions
    """
    # create a dataframe containing people whose interactions >= 5
    users_interactions_cnt_df = interactions_df.groupby(
        ["personId", "contentId"]).size().groupby("personId").size()
    print("# of users ", len(users_interactions_cnt_df))
    users_enough_interactions_cnt_df = users_interactions_cnt_df[
        users_interactions_cnt_df >= 5].reset_index()[["personId"]]
    print("# of users with enough interactions ",
          len(users_enough_interactions_cnt_df))

    # create a dataframe from interactions_df which are involved with selected people
    print("# of total interactions ", len(interactions_df))
    interactions_from_selected_users_df = interactions_df.merge(
        users_enough_interactions_cnt_df, how="right", left_on="personId",
        right_on="personId")

    print('# of interactions from users with at least 5 interactions: %d' %
          len(interactions_from_selected_users_df))
    """
    aggregate # of interactions
    """
    interactions_full_df = interactions_from_selected_users_df.groupby(
        ["personId", "contentId"])["eventStrength"].sum().apply(
            smooth_user_preference).reset_index()
    print("# of unique user/item interactions: {}".format(len(interactions_full_df)))

    interactions_train_df, interactions_test_df = train_test_split(
        interactions_full_df, stratify=interactions_full_df["personId"],
        test_size=0.20, random_state=1)
    print("# of interactions on Train set: {}".format(len(interactions_train_df)))
    print("# of interactions on Test set: {}".format(len(interactions_test_df)))

    # indexing by personId to speed up the searches during evaluation
    interactions_full_indexed_df = interactions_full_df.set_index("personId")
    interactions_train_indexed_df = interactions_train_df.set_index("personId")
    interactions_test_indexed_df = interactions_test_df.set_index("personId")

    return interactions_full_indexed_df, interactions_train_indexed_df,\
        interactions_test_indexed_df, interactions_full_df, interactions_train_df



# aggregate number of interactions between personId and contentId
def smooth_user_preference(x):
    return math.log(1 + x, 2)

# get item associated with person


def get_items_interacted(person_id, interactions_indexed_df):
    interacted_items = interactions_indexed_df.loc[person_id]["contentId"]
    return set(interacted_items if type(interacted_items) == pd.Series
               else [interacted_items])


def getItemPopularityDf(interactions_indexed_df):
    item_popularity_df = interactions_indexed_df.groupby(
        "contentId")["eventStrength"].sum().sort_values(
            ascending=False).reset_index()

    return item_popularity_df


def getStopWords(tps=['english', 'portuguese']):
    stopwords_list = []
    for tp in tps:
        stopwords_list += stopwords.words(tp)

    return stopwords_list


def getVectorizer(tp, stopwords_list):
    if tp == "tfidf":
        vectorizer = TfidfVectorizer(analyzer="word",
                                     ngram_range=(1, 2),
                                     min_df=0.003,
                                     max_df=0.5,
                                     max_features=5000,
                                     stop_words=stopwords_list)

    return vectorizer


def getMatrix(data, vectorizer_type="tfidf"):
    stopwords_list = getStopWords()
    vectorizer = getVectorizer(
        vectorizer_type, stopwords_list)
    matrix = vectorizer.fit_transform(data)

    return matrix


def getPredictionsDfFromSVD(data, number_of_factors):
    users_items_pivot_matrix_df = data.pivot(index="personId",
                                             columns="contentId",
                                             values="eventStrength"
                                             ).fillna(0)
    users_items_pivot_matrix = users_items_pivot_matrix_df.values
    users_ids = list(users_items_pivot_matrix_df.index)

    U, sigma, Vt = svds(users_items_pivot_matrix, k=number_of_factors)
    sigma = np.diag(sigma)
    all_users_to_items_predictions_ratings = np.dot(np.dot(U, sigma), Vt)

    cf_preds_df = pd.DataFrame(all_users_to_items_predictions_ratings,
                               columns=users_items_pivot_matrix_df.columns,
                               index=users_ids).transpose()

    return cf_preds_df
