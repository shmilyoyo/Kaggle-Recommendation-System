import pandas as pd
import preprocessData
from popularityModel import PopularityModel
from evaluate import ModelEvaluator
from profile import Profile
from contentBasedModel import CBRecommender
from collaborativeFilteringBasedModel import CFRecommender
from hybridRecommender import HybridRecommender
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import pickle
import json

from pathlib import Path

from ldaTopicModel import LdaTopicModel

inputDataRootPath = "/home/xuhao/Data/Kaggle-Recommendation-Dataset"
outputDataRootPath = "/home/xuhao/Data/Kaggle-Recommendation-Dataset"

articles_df = pd.read_csv(inputDataRootPath + "/" + "shared_articles.csv")
articles_df = articles_df[articles_df['eventType'] == "CONTENT SHARED"]
articles_df = articles_df[articles_df['lang'] == 'en']
print(len(articles_df))

# test preprocessData Module
interactions_full_indexed_df, interactions_train_indexed_df, \
    interactions_test_indexed_df, interactions_full_df, \
    interactions_train_df = preprocessData.mungingData(inputDataRootPath,
                                                       outputDataRootPath)
"""
# person_id = -8845298781299428018
# print(preprocessData.get_items_interacted(person_id, interactions_full_indexed_df))

# model evaluator
model_evaluator = ModelEvaluator(articles_df, interactions_test_indexed_df,
                                 interactions_train_indexed_df,
                                 interactions_test_indexed_df,
                                 )

# test PopularityModel Module
item_popularity_df = preprocessData.getItemPopularityDf(
    interactions_full_indexed_df)

popularity_model = PopularityModel(
    "popularity", item_popularity_df, articles_df)

print("Evaluating Popularity Recommendation Model...")
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(
    popularity_model)
print("\nGlobal metrics: \n {}".format(pop_global_metrics))

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(pop_detailed_results_df.head(10))

# test Profile Module
item_ids = articles_df['contentId'].tolist()
tfidf_matrix = preprocessData.getMatrix(
    articles_df["title"] + "" + articles_df["text"])

content_ids = articles_df['contentId'].tolist()

profile = Profile(content_ids, tfidf_matrix, interactions_full_df, articles_df)
users_profiles = profile.build_users_profiles()

# test Content-based Module
cb_recommender = CBRecommender(
    "content-based", item_ids, users_profiles, tfidf_matrix)

print("Evaluating Content-Based Filtering Model...")
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(
    cb_recommender)
print("\nGlobal Metrics: \n {}".format(cb_global_metrics))

# test Collaborative Filtering based Module
cf_predictions_df = preprocessData.getPredictionsDfFromSVD(
    interactions_train_df, 15)
cf_recommender = CFRecommender(
    "Collaborative Filtering Based Model", cf_predictions_df, articles_df)

print("Evaluating Collaborative Filtering (SVD Matrix Factorization) model...")
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(
    cf_recommender)
print("\nGlobal metrics: \n{}".format(cf_global_metrics))

# test Hybrid Module
hb_recommender = HybridRecommender(
    "Hybrid Based Model", cb_recommender, cf_recommender, articles_df)

print('Evaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(
    hb_recommender)
print('\nGlobal metrics:\n%s' % hybrid_global_metrics)


# present the every model results
total_metrics = [pop_global_metrics, cf_global_metrics,
                 cb_global_metrics, hybrid_global_metrics]
global_metrics_df = pd.DataFrame(total_metrics).set_index("modelName")
global_metrics_df.to_pickle(outputDataRootPath + "/" + "global_metrics_df.pkl")
"""

# global_metrics_df = pd.read_pickle(outputDataRootPath + "/" + "global_metrics_df.pkl")
# ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15, 8))
# for p in ax.patches:
#     ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')
# fig = ax.get_figure()
# fig.savefig(outputDataRootPath + "/" + "global_metrics_df.pdf")

corpus = articles_df.text.tolist()
model_path = "/home/xuhao/Library/mallet-2.0.8/bin/mallet"
ldaTm = LdaTopicModel("LDA_Topic_Model", outputDataRootPath,
                      model_type="mallet", model_path=model_path)
print("Preprocessing Data...")
ldaTm.preprocess_data(corpus)
print("Training Data...")
ldaTm.get_best_model(24, 2, 2)

with (Path(outputDataRootPath) / 'mallet_model/corpus_bow.pkl').open("rb") as fp:
    corpus_bow = pickle.load(fp)

docs_ids = articles_df["contentId"].tolist()

topics_docs_full_df = ldaTm.get_topics_docs_full_df(
    corpus_bow, docs_ids, interactions_full_df)

# print(set(articles_df[articles_df['contentId'].isin(topics_docs_full_df['Documnet_Id'])]['lang']))
# print(set(topics_docs_full_df['Documnet_Id']) == set(articles_df['contentId']))

topics_to_cnt = ldaTm.get_topics_to_cnt_for_doc_ids(topics_docs_full_df, docs_ids)
print(topics_to_cnt)