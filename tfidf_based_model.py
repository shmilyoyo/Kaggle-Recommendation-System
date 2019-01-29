from base_model import BaseModel
from content_based_model import ContentedBasedModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy
import scipy
import preprocess_data
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sklearn
import pickle
from pathlib import Path


class TfidfBasedModel(ContentedBasedModel):

    MODEL_NAME = "Tfidf-Based Model"

    def __init__(self, model_id, outputRootPath):
        """Initilaize the data used in this classself.

        Arguments:
            model_id {int} - the model id
            contents_ids {DataFrame} -- contents ids in original articles
            users_profiles {dict} -- mapping person_id to profile vector
            matrix {matrix} -- the matrix returned from vectorizer
            items_df {DataFrame} -- the default additional items added in result (default: {None})
        """

        self.model_id = model_id
        self.outputRootPath = Path(outputRootPath)
        self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.model_type = "tfidf"

    def get_model_name(self):
        return self.MODEL_NAME

    def runModel(self):
        pass

    def generate_profile(self, user_id, articles_df, interactions_full_df):
        pass

    def _analyzer(self, docs):
        docs = preprocess_data.remove_email(docs)
        docs = preprocess_data.remove_newline(docs)
        docs = preprocess_data.remove_single_quote(docs)
        docs_words = list(preprocess_data.doc_to_words(docs))

        docs_words = preprocess_data.remove_stopwords(docs_words)
        docs_words = preprocess_data.lemmatized(
            self.nlp, docs_words, allowed_postags=['NOUN', 'ADJ', 'ADV'])
        docs_words = preprocess_data.remove_stopwords(docs_words)

        return docs_words

    def train_model(self, docs):
        outputFolderPath = self.outputRootPath / \
            (self.model_type + "_model")

        if not outputFolderPath.exists():
            outputFolderPath.mkdir()

        vectorizer = TfidfVectorizer(self._analyzer,
                                     ngram_range=(1, 2),
                                     min_df=0.003,
                                     max_df=0.5,
                                     max_features=5000)

        self.model = vectorizer.fit(docs)
        # with (outputFolderPath / "model.pkl").open("wb") as fp:
        #     pickle.dump(vectorizer, fp)
        # self.model = model

    def _get_embedding(self, doc):
        embedding = self.model.transform([doc])

        return embedding

    def _get_item_profile(self, doc):
        """Get the vector corresponding to doc in matrix.

        Arguments:
            doc {int} -- the item id in matrix

        Returns:
            np.array -- the vector in the matrix
        """

        item_profile = self._get_embedding(doc)

        return item_profile

    def get_items_profiles(self, docs):
        """Get a list of vectors corresponding to docs in matrix.

        Arguments:
            docs {list} -- a list of ids need to be got from matrix

        Returns:
            sparse matrix -- the total vectors corresponding to docs
        """

        items_ids, items_contents = zip(*docs)

        items_profiles_list = [self._get_item_profile(
            item_content) for item_content in items_contents]
        items_profiles = scipy.sparse.vstack(items_profiles_list)

        return items_ids, items_profiles

    def get_user_profile(self, user_id, articles_df, interactions_full_df):
        """Build a user profile based on the normalization.

        normalize the items profiles for person with user_id based on the content strength

        Arguments:
            user_id {int} -- the person index
            interactions_indexed_df {DataFrame} -- interactions DataFrame indexed on personId

        Returns:
            matrix -- a user normalized profile
        """
        outputFolderPath = self.outputRootPath / \
            (self.model_type + "_model") / "profiles"
        if not outputFolderPath.exists():
            outputFolderPath.mkdir()

        interactions_person_df = interactions_full_df[interactions_full_df['personId'] == user_id]

        contentIds = interactions_person_df['contentId'].tolist()
        items_df = articles_df[articles_df['contentId'].isin(contentIds)]
        items_dict = pd.Series(
            items_df['text'].values, index=items_df['contentId']).to_dict()

        items_ids, items_profiles = self.get_items_profiles(items_dict.items())

        # average weight the matrix
        user_items_strengths = []
        for item_id in items_ids:
            user_items_strengths += interactions_person_df[interactions_person_df['contentId']
                                                           == item_id]['eventStrength'].tolist()
        user_items_strengths = np.array(user_items_strengths).reshape(-1, 1)

        user_items_strengths_weighted_avg = np.sum(items_profiles.multiply(
            user_items_strengths), axis=0) / np.sum(user_items_strengths)
        user_profile_strength_norm = sklearn.preprocessing.normalize(
            user_items_strengths_weighted_avg)
        with (outputFolderPath / (str(user_id) + "_strength_norm.pkl")).open("wb") as fp:
            pickle.dump(user_profile_strength_norm, fp)

        return user_profile_strength_norm

    def load_user_profile(self, user_id):
        """Load the profile for user_id.

        Arguments:
            user_id {int} -- the user id.

        Returns:
            tuple -- (topic_id -> number of related docs,
                      dict -- topic_id -> strength)
        """

        outputFolderPath = self.outputRootPath / \
            (self.model_type + "_model") / "profiles"
        with (outputFolderPath / (str(user_id) + "_strength_norm.pkl")).open("rb") as fp:
            user_profile_strength_norm = pickle.load(fp)

        return user_profile_strength_norm

    def get_score_of_docs(self, user_id, docs):
        """Get similar items to the user profile based on cosine similarityself.

        Arguments:
            person_id {int} -- the person id

        Keyword Arguments:
            topn {int} -- the number of most similar items (default: {1000})

        Returns:
            list -- a ranked list of most similar items to the user profile
        """
        user_profile_strength_norm = self.load_user_profile(user_id)

        items_ids, items_profiles = self.get_items_profiles(docs)

        cosine_similarities = cosine_similarity(
            user_profile_strength_norm, items_profiles)

        item_id_to_strength_weight_score = [(items_ids[i], cosine_similarities[0, i])
                                            for i in range(cosine_similarities.shape[1])]

        return item_id_to_strength_weight_score

    def recommend_items(self, person_id, docs, articles_df, items_to_ignore=[], topn=10,
                        verbose=False):
        """Recommend items to person with person_idself.

        Arguments:
            person_id {int} -- person id

        Keyword Arguments:
            items_to_ignore {list} -- a list of items that user has already visited (default: {[]})
            topn {int} -- the number of final recommendation (default: {10})
            verbose {bool} -- indicate whether add items extra information (default: {False})

        Raises:
            Exception -- if want to use verbose mode, the items_df should be provided

        Returns:
            DataFrame -- a recommendation dataframe
        """

        item_id_to_strength_weight_score = self.get_score_of_docs(person_id, docs)
        # filter out the items that are already interacted
        item_id_to_strength_weight_score.sort(key=lambda x: x[1], reverse=True)
        similar_items_filter = list(
            filter(lambda x: x[0] not in items_to_ignore, item_id_to_strength_weight_score))

        recommendations_df = pd.DataFrame(similar_items_filter, columns=[
                                          "contentId", "recStrength"]).head(topn)

        if verbose:
            if articles_df is None:
                raise Exception("articles_df is required in verbose mode.")

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
