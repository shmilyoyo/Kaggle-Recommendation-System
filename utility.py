import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from pathlib import Path
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import re


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


def get_texts_for_user(user_id, articles_df, interactions_df):
    contentIds = interactions_df.groupby('personId')['contentId'].tolist()
    texts = articles_df[articles_df['contentId'].isin(
        contentIds)]['text'].tolist()

    return texts

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


def remove_email(docs):
    return [re.sub(r'\S*@\S*\s?', '', doc) for doc in docs]


def remove_newline(docs):
    return [re.sub(r'\s+', ' ', doc) for doc in docs]


def remove_single_quote(docs):
    return [re.sub("\'", "", doc) for doc in docs]


def doc_to_words(docs):
    for doc in docs:
        yield(simple_preprocess(str(doc), deacc=True))


def build_n_gram(docs_words):
    bigram = gensim.models.Phrases(docs_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[docs_words], threshold=100)

    return bigram, trigram


def remove_stopwords(docs_words):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', "things",
                       "that's", "something", "take", "don't", "may",
                       "want", "you're", "set", "might", "says",
                       "including", "lot", "much", "said", "know", "good",
                       "step", "often", "going", "thing", "things", "think",
                       "back", "actually", "better", "look", "find", "right",
                       "example", "verb", "verbs"])

    return [[word for word in doc if word not in set(stop_words)] for doc in docs_words]


def make_bigrams(model, docs_words):
    return [model[doc] for doc in docs_words]


def make_trigrams(model, docs_words):
    return [model[doc] for doc in docs_words]


def lemmatized(nlp, docs_words, allowed_postags=['NOUN', 'ADJ', 'ADV']):
    # texts are list of lists of words
    texts_out = []
    for doc in docs_words:
        doc = nlp(" ".join(doc))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def get_weight(user_id, interactions_full_df, items_ids):
    # average weight the matrix
    interactions_person_df = interactions_full_df[interactions_full_df['personId'] == user_id]
    user_items_strengths = []
    for item_id in items_ids:
        user_items_strengths += interactions_person_df[interactions_person_df['contentId']
                                                        == item_id]['eventStrength'].tolist()
    return user_items_strengths

def get_items(user_id, interactions_full_df, articles_df):
    interactions_person_df = interactions_full_df[interactions_full_df['personId'] == user_id]

    contentIds = interactions_person_df['contentId'].tolist()
    items_df = articles_df[articles_df['contentId'].isin(contentIds)]
    items_dict = pd.Series(
        items_df['text'].values, index=items_df['contentId']).to_dict()
    items_ids, items_contents = zip(*list(items_dict.items()))
    return items_ids, items_contents