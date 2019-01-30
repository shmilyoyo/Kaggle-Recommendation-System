from base_model import BaseModel
from content_based_model import ContentedBasedModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy
import scipy
import utility
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
            model_id {int} - the model id.
            contents_ids {DataFrame} -- contents ids in original articles.
            users_profiles {dict} -- mapping person_id to profile vector.
            matrix {matrix} -- the matrix returned from vectorizer.
            items_df {DataFrame} -- the default additional items added in result (default: {None}).
        """

        self.model_id = model_id
        self.outputRootPath = Path(outputRootPath)
        self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.model_type = "tfidf"
        self.model = None
        self.items_profiles = None

    def get_model_name(self):
        return self.MODEL_NAME

    def runModel(self):
        pass

    def _preprocess_data(self, docs):
        """Self defined analyzer for TfidfVectorizer.
        
        Arguments:
            docs {list} -- a list of doc text.
        
        Returns:
            list -- a list of lists of doc text with word tokens.
        """

        docs = utility.remove_email(docs)
        docs = utility.remove_newline(docs)
        docs = utility.remove_single_quote(docs)
        docs_words = list(utility.doc_to_words(docs))

        docs_words = utility.remove_stopwords(docs_words)
        docs_words = utility.lemmatized(
            self.nlp, docs_words, allowed_postags=['NOUN', 'ADJ', 'ADV'])
        docs_words = utility.remove_stopwords(docs_words)

        docs = [" ".join(doc_word) for doc_word in docs_words]
        return docs

    def train_model(self, docs):
        """Train the model.
        
        Arguments:
            docs {list} -- a list of doc text.
        """

        outputFolderPath = self.outputRootPath / \
            (self.model_type + "_model")

        if not outputFolderPath.exists():
            outputFolderPath.mkdir()

        docs = self._preprocess_data(docs)
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

    def _get_embedding(self, doc):
        """Get embedding representation for doc text.
        
        Arguments:
            doc {str} -- the doc text.
        
        Returns:
            matrix -- the embedding of the doc text.
        """

        embedding = self.model.transform([doc])

        return embedding

    def _get_item_profile(self, doc):
        """Get the vector corresponding to doc in matrix.

        Arguments:
            doc {str} -- the doc text.

        Returns:
            np.array -- the vector in the matrix.
        """

        item_profile = self._get_embedding(doc)

        return item_profile

    def get_items_profiles(self, docs):
        """Get a list of vectors corresponding to docs in matrix.

        Arguments:
            docs {list} -- a list of doc text.

        Returns:
            sparse matrix -- the total vectors corresponding to docs.
        """

        items_contents = self._preprocess_data(docs)

        items_profiles_list = [self._get_item_profile(
            item_content) for item_content in items_contents]
        items_profiles = scipy.sparse.vstack(items_profiles_list)

        return items_profiles

    def load_items_profiles(self, items_ids, articles_df):
        outputFolderPath = self.outputRootPath / (self.model_type + "_model")

        if (outputFolderPath / "items_profiles.pkl").exists():
            with (outputFolderPath / "items_profiles.pkl").open("rb") as fp:
                items_profiles = pickle.load(fp)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(outputFolderPath / "items_profiles.pkl"))

        items_profiles_indexes = []
        for item_id in items_ids:
            index = articles_df[articles_df['contentId']
                                == item_id].index.tolist()
            items_profiles_indexes += index

        items_profiles_list = [items_profiles[item_profile_index] for item_profile_index in items_profiles_indexes]
        items_profiles = scipy.sparse.vstack(items_profiles_list)

        return items_profiles

    def get_user_profile(self, user_id, interactions_full_df, articles_df):
        """Build a user profile based on the normalization.

        normalize the items profiles for person with user_id based on the content strength.

        Arguments:
            user_id {int} -- the person index.
            interactions_full_df {DataFrame} -- interactions DataFrame.
            articles_df {DataFrame} -- articles DataFrame.

        Returns:
            matrix -- a user normalized profile.
        """
        outputFolderPath = self.outputRootPath / \
            (self.model_type + "_model") / "profiles"
        if not outputFolderPath.exists():
            outputFolderPath.mkdir()

        items_ids = utility.get_items_ids(
            user_id, interactions_full_df, articles_df)

        items_profiles = self.load_items_profiles(items_ids, articles_df)
        print(items_profiles.shape)

        # average weight the matrix
        user_items_strengths = utility.get_weight(
            user_id, interactions_full_df, items_ids)

        user_items_strengths = np.array(user_items_strengths).reshape(-1, 1)
        print(user_items_strengths.shape)

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
            matrix -- the user_id profile.
        """

        outputFolderPath = self.outputRootPath / \
            (self.model_type + "_model") / "profiles"
            
        with (outputFolderPath / (str(user_id) + "_strength_norm.pkl")).open("rb") as fp:
            user_profile_strength_norm = pickle.load(fp)

        return user_profile_strength_norm

    def get_score_of_docs(self, user_id, docs):
        """Get similar items to the user profile based on cosine similarityself.

        Arguments:
            user_id {int} -- the user id.
            docs {list} -- a list of docs with ids: [(id, content), ...].

        Returns:
            list -- a list of doc id to score: [(id, score)].
        """
        user_profile_strength_norm = self.load_user_profile(user_id)

        items_ids, items_content = zip(*docs)
        items_profiles = self.get_items_profiles(items_content)

        cosine_similarities = cosine_similarity(
            user_profile_strength_norm, items_profiles)

        item_id_to_strength_weight_score = [(items_ids[i], cosine_similarities[0, i])
                                            for i in range(cosine_similarities.shape[1])]

        return item_id_to_strength_weight_score

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
