import pandas as pd
import preprocessData
from popularityModel import PopularityModel
from evaluate import ModelEvaluator
from profile import Profile
from contentBasedModel import ContentBasedModel
from collaborativeFilteringBasedModel import CFRecommender

inputDataRootPath = "/data/haoxu/Data/Kaggle-Recommendation-Dataset"
outputDataRootPath = "/data/haoxu/Data/Kaggle-Recommendation-Dataset"

articles_df = pd.read_csv(inputDataRootPath + "/" + "shared_articles.csv")
articles_df = articles_df[articles_df['eventType'] == "CONTENT SHARED"]

# test preprocessData Module
interactions_full_indexed_df, interactions_train_indexed_df, \
    interactions_test_indexed_df, interactions_full_df, \
    interactions_train_df = preprocessData.mungingData(inputDataRootPath,
                                                       outputDataRootPath)

# person_id = -8845298781299428018
# print(preprocessData.get_items_interacted(person_id, interactions_full_indexed_df))

# model evaluator
model_evaluator = ModelEvaluator(articles_df, interactions_test_indexed_df,
                                 interactions_train_indexed_df,
                                 interactions_test_indexed_df,
                                 )

"""# test PopularityModel Module
item_popularity_df = preprocessData.getItemPopularityDf(
    interactions_full_indexed_df)

popularity_model = PopularityModel(
    "popularity", item_popularity_df, articles_df)

print("Evaluating Popularity Recommendation Model...")
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(
    popularity_model)
print("Global metrics: \n {}".format(pop_global_metrics))

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(pop_detailed_results_df.head(10))"""

"""# test Profile Module
item_ids = articles_df['contentId'].tolist()
tfidf_matrix = preprocessData.getMatrix(
    articles_df["title"] + "" + articles_df["text"])

content_ids = articles_df['contentId'].tolist()

profile = Profile(content_ids, tfidf_matrix, interactions_full_df, articles_df)
users_profiles = profile.build_users_profiles()

# test Content-based Module
content_based_model = ContentBasedModel(
    "content-based", item_ids, users_profiles, tfidf_matrix)

print("Evaluating Content-Based Filtering Model...")
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_model)
print("Global Metrics: \n {}".format(cb_global_metrics))"""

# test Collaborative Filtering based Module
cf_predictions_df = preprocessData.getPredictionsDfFromSVD(
    interactions_train_df, 15)
cf_recommender_model = CFRecommender(
    "Collaborative Filtering Based Model", cf_predictions_df, articles_df)

print("Evaluating Collaborative Filtering (SVD Matrix Factorization) model...")
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print("Global metrics: \n{}".format(cf_global_metrics))