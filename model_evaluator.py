# EVALUATION
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

import utility
import random
import pandas as pd


class ModelEvaluator:
    """Evaluate Model Module based on Recall@N.
    """

    def __init__(self, articles_df, interactions_full_indexed_df,
                 interactions_train_indexed_df, interactions_test_indexed_df):
        """Initialize the basic data frameself.

        Arguments:
            articles_df {pd.DataFrame} -- the original whole DataFrame
            interactions_full_indexed_df {pd.DataFrame} -- full interactions DataFrame
            interactions_train_indexed_df {pd.DataFrame} -- training interactions DataFrame
            interactions_test_indexed_df {pd.DataFrame} -- testing interactions DataFrame
        """

        self.articles_df = articles_df
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.interactions_test_indexed_df = interactions_test_indexed_df

    def get_not_interacted_item_samples(self, person_id,
                                        sample_size, seed=1):
        """Get irrelevant items sample.

        Arguments:
            person_id {int} -- the person id
            sample_size {int} -- the sample size

        Keyword Arguments:
            seed {int} -- random seed (default: {1})

        Returns:
            set -- irrelevant items samples
        """

        interacted_items = utility.get_items_interacted(
            person_id, self.interactions_full_indexed_df)
        all_items = set(self.articles_df["contentId"])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_samples = random.sample(
            non_interacted_items, sample_size)

        return set(non_interacted_items_samples)

    def _verify_hit_top_n(self, item_id, recommend_items, topn):
        """Verify whether item_id among recommended is in topnself.

        Arguments:
            item_id {int} -- test item id
            recommend_items {pd.DataFrame} -- ranking list returned by model
            topn {int} -- top n

        Returns:
            tuple -- (whether hit in topn, the index of hitted item among ranking list)
        """

        try:
            index = next(i for i, c in enumerate(
                recommend_items) if c == item_id)
        except:
            index = - 1

        hit = int(index in range(topn))

        return hit, index

    def evaluate_model_for_user(self, model, person_id):
        """Evaluate the model for person with person_id

        Arguments:
            model {model} -- the recommendation model
            person_id {int} -- the person id

        Returns:
            dict -- the metrics for person with person_id
        """

        interacted_value_testset = self.interactions_test_indexed_df.loc[person_id]
        if type(interacted_value_testset["contentId"]) == pd.Series:
            person_interacted_item_testset = set(
                interacted_value_testset["contentId"])
        else:
            person_interacted_item_testset = set(
                [int(interacted_value_testset["contentId"])])
        interacted_items_count_testset = len(person_interacted_item_testset)

        # get a ranked recommendation list form a model for a given user
        person_recs_df = model.recommend_items(person_id,
                                               items_to_ignore=utility.get_items_interacted(
                                                   person_id,
                                                   self.interactions_train_indexed_df),
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        # for each item the user has interacted in test set
        for item_id in person_interacted_item_testset:
            non_interacted_items_samples = self.get_not_interacted_item_samples(
                person_id,
                sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                seed=item_id % (2 ** 32))
            # combining the current interacted item with the 100 random samples
            item_to_filter_recs = non_interacted_items_samples.union(
                set([item_id]))
            # filtering recommendations that are from item_to_filter_recs
            valid_recs_df = person_recs_df[person_recs_df["contentId"].isin(
                item_to_filter_recs)]
            valid_recs = valid_recs_df["contentId"].values

            # verifying if current interacted item is among the Top-N recommended items
            hits_at_5, index_at_5 = self._verify_hit_top_n(
                item_id, valid_recs, 5)
            hits_at_5_count += hits_at_5
            hits_at_10, index_at_10 = self._verify_hit_top_n(
                item_id, valid_recs, 10)
            hits_at_10_count += hits_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items, when mixed with a set of non-relevant items.
        recall_at_5 = hits_at_5_count / interacted_items_count_testset
        recall_at_10 = hits_at_10_count / interacted_items_count_testset

        person_metrics = {"hits@5_count": hits_at_5_count,
                          "hits@10_count": hits_at_10_count,
                          "recall@5": recall_at_5,
                          "recall@10": recall_at_10,
                          "interacted_count": interacted_items_count_testset}

        return person_metrics

    def evaluate_model(self, model):
        """Evaluate the modelself.
        
        Arguments:
            model {model} -- the recommendation model
        
        Returns:
            tuple -- (the global metrics, the whole people metrics DataFrame)
        """

        people_metrics = []
        for idx, person_id in enumerate(list(
                self.interactions_test_indexed_df.index.unique().values)):
            person_metrics = self.evaluate_model_for_user(model, person_id)
            person_metrics["_person_id"] = person_id
            people_metrics.append(person_metrics)

        print("%d users processed" % idx)

        people_metrics_df = pd.DataFrame(people_metrics).sort_values(
            'interacted_count', ascending=False)

        # global metrics
        global_recall_at_5 = people_metrics_df["hits@5_count"].sum(
        ) / people_metrics_df["interacted_count"].sum()
        global_recall_at_10 = people_metrics_df["hits@10_count"].sum(
        ) / people_metrics_df["interacted_count"].sum()

        global_metrics = {"modelName": model.get_model_name(),
                          "recall@5": global_recall_at_5,
                          "recall@10": global_recall_at_10}

        return global_metrics, people_metrics_df
