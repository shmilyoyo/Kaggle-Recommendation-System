import errno
import os
import pickle

import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from content_based_model import ContentedBasedModel


class TfidfBasedModel(ContentedBasedModel):

    def __init__(self, model_id, output_root_path):
        """Initilaize the data used in this class.

        Arguments:
            model_id {int} - the model id.
            output_root_path {str} -- the output root path.
        """
        super().__init__(model_id, output_root_path)
        self.model_type = "tfidf"
        self.items_profiles = None

    def get_model_name(self):
        """
        Get model name.
        
        Returns:
            str -- the model id
        """

        return self.model_id

    def runModel(self):
        pass

    def train_model(self, docs):
        """
        Train the model.
        
        Arguments:
            docs {list} -- a list of doc text.
        """

        outputFolderPath = self.output_root_path / \
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

    def load_model(self):
        """
        Load the model.
        
        Raises:
            FileNotFoundError -- raise the error when model is not in path.
        """

        outputFolderPath = self.output_root_path / (self.model_type + "_model")

        if (outputFolderPath / "model.pkl").exists():
            with (outputFolderPath / "model.pkl").open("rb") as fp:
                self.model = pickle.load(fp)
                print("model loaded from {}".format(str(outputFolderPath / "model.pkl")))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(outputFolderPath / "model.pkl"))

    def get_embedding(self, doc):
        """
        Get embedding representation for doc text.
        
        Arguments:
            doc {str} -- the doc text.
        
        Returns:
            matrix -- the embedding of the doc text.
        """

        embedding = self.model.transform([doc])

        return embedding

    def get_items_profiles(self, docs):
        """
        Get a list of vectors corresponding to docs in matrix.

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
        """
        Load items profiles based on items ids.
        
        Arguments:
            items_ids {list} -- a list of items ids.
            articles_df {pandas.DataFrame} -- the articles dataframe.
        
        Returns:
            sparse matrix -- the items profiles sparse matrix.
        """

        items_profiles_list = super().load_items_profiles(items_ids, articles_df)
        items_profiles = scipy.sparse.vstack(items_profiles_list)

        return items_profiles