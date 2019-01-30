import errno
import os
import pickle

import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from content_based_model import ContentedBasedModel


class TfidfBasedModel(ContentedBasedModel):

    MODEL_NAME = "Tfidf-Based Model"

    def __init__(self, model_id, outputRootPath):
        """Initilaize the data used in this classself.

        Arguments:
            model_id {int} - the model id.
            contents_ids {DataFrame} -- contents ids in original articles.
            users_profiles {dict} -- mapping person_id to profile vector.
            matrix {matrix} -- the matrix returned from vectorizer.
            items_df {DataFrame} -- the default additional items added in result (default: {None}).
        """
        super().__init__(model_id, outputRootPath)
        self.model_type = "tfidf"
        self.items_profiles = None

    def get_model_name(self):
        return self.model_id

    def runModel(self):
        pass

    def train_model(self, docs):
        """Train the model.
        
        Arguments:
            docs {list} -- a list of doc text.
        """

        outputFolderPath = self.outputRootPath / \
            (self.model_type + "_model")

        if not outputFolderPath.exists():
            outputFolderPath.mkdir()

        docs = self.preprocess_data(docs)
        vectorizer = TfidfVectorizer(analyzer='word',
                                     ngram_range=(1, 2),
                                     min_df=0.003,
                                     max_df=0.5,
                                     max_features=5000)

        items_profiles = vectorizer.fit_transform(docs)
        with (outputFolderPath / "items_profiles.pkl").open("wb") as fp:
            pickle.dump(items_profiles, fp)

        self.model = vectorizer
        with (outputFolderPath / "model.pkl").open("wb") as fp:
            pickle.dump(self.model, fp)
        # self.model = model

    def load_model(self):
        outputFolderPath = self.outputRootPath / (self.model_type + "_model")

        if (outputFolderPath / "model.pkl").exists():
            with (outputFolderPath / "model.pkl").open("rb") as fp:
                self.model = pickle.load(fp)
                print("model loaded from {}".format(str(outputFolderPath / "model.pkl")))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(outputFolderPath / "model.pkl"))

    def get_embedding(self, doc):
        """Get embedding representation for doc text.
        
        Arguments:
            doc {str} -- the doc text.
        
        Returns:
            matrix -- the embedding of the doc text.
        """

        embedding = self.model.transform([doc])

        return embedding

    def get_items_profiles(self, docs):
        """Get a list of vectors corresponding to docs in matrix.

        Arguments:
            docs {list} -- a list of doc text.

        Returns:
            sparse matrix -- the total vectors corresponding to docs.
        """

        items_contents = self.preprocess_data(docs)

        items_profiles_list = [self.get_item_profile(
            item_content) for item_content in items_contents]
        items_profiles = scipy.sparse.vstack(items_profiles_list)

        return items_profiles

    def load_items_profiles(self, items_ids, articles_df):
        items_profiles_list = super().load_items_profiles(items_ids, articles_df)
        items_profiles = scipy.sparse.vstack(items_profiles_list)

        return items_profiles

    def recommend_items(self, person_id, docs, articles_df, items_to_ignore=[], topn=10):
        """Recommend items to person with person_idself.

        Arguments:
            person_id {int} -- person id.
            docs {list} -- a list of docs with ids: [(id, content), ...].
        Keyword Arguments:
            items_to_ignore {list} -- a list of items that user has already visited (default: {[]}).
            topn {int} -- the number of final recommendation (default: {10}).
            verbose {bool} -- indicate whether add items extra information (default: {False}).

        Raises:
            Exception -- if want to use verbose mode, the items_df should be provided.

        Returns:
            DataFrame -- a recommendation dataframe.
        """

        item_id_to_strength_weight_score = self.get_score_of_docs(
            person_id, docs)
        # filter out the items that are already interacted
        item_id_to_strength_weight_score.sort(key=lambda x: x[1], reverse=True)
        similar_items_filter = list(
            filter(lambda x: x[0] not in items_to_ignore, item_id_to_strength_weight_score))

        recommendations_df = pd.DataFrame(similar_items_filter, columns=[
                                          "contentId", "recStrength"]).head(topn)

        recommendations_df = recommendations_df.merge(articles_df,
                                                          how="left",
                                                          left_on="contentId",
                                                          right_on="contentId"
                                                          )[["recStrength",
                                                             "contentId",
                                                             "title",
                                                             "url",
                                                             "lang"]]
        return recommendations_df
