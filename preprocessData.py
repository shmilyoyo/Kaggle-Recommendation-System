# DATA MUNGING

import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from sklearn.model_selection import train_test_split
from pathlib import Path


def mungingData(inputDataRootPath, outputDataRootPath):
    inputDataRootPath = Path(inputDataRootPath)
    outputDataRootPath = Path(outputDataRootPath)

    # load data
    articles_df = pd.read_csv(str(inputDataRootPath / "shared_articles.csv"))
    articles_df = articles_df[articles_df['eventType']
                              == "CONTENT SHARED"]

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

    """
    aggregate # of interactions
    """
    interactions_full_df = interactions_from_selected_users_df.groupby(
        ["personId", "contentId"])["eventStrength"].sum().apply(
            smooth_user_preference).reset_index()
    print("# of unique user/item interactions: {}".format(len(interactions_full_df)))

    interactions_train_df, interactions_test_df = train_test_split(
        interactions_full_df, stratify=interactions_full_df["personId"],
        test_size=0.2, random_state=1)
    print("# of interactions on Train set: {}".format(len(interactions_train_df)))
    print("# of interactions on Test set: {}".format(len(interactions_test_df)))

    # indexing by personId to speed up the searches during evaluation
    interactions_full_indexed_df = interactions_full_df.set_index("personId")
    interactions_train_indexed_df = interactions_train_df.set_index("personId")
    interactions_test_indexed_df = interactions_test_df.set_index("personId")

    return interactions_full_indexed_df, interactions_train_indexed_df,\
        interactions_test_indexed_df


# aggregate number of interactions between personId and contentId
def smooth_user_preference(x):
    return math.log(1 + x, 2)

# get item associated with person


def get_items_interacted(person_id, interactions_df):
    interacted_items = interactions_df.loc[person_id]["contentId"]
    return set(interacted_items if type(interacted_items) == pd.Series
               else [interacted_items])
