from base_model import BaseModel
import utility

import re
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

# Gensim
import gensim
from gensim.models import CoherenceModel
import gensim.corpora as corpora

# spacy for lemmatization
import spacy

from pathlib import Path
import pickle
import json
import sklearn
from sklearn.metrics.pairwise import cosine_similarity



class LdaTopicModel(BaseModel):
    """Recommendation model based on LDA topic model.

    Arguments:
        BaseModel {abstract class} -- the base abstract class.
    """

    def __init__(self, model_id, outputRootPath, model_type='default', model_path=None):
        """Initialize the parameters.

        Arguments:
            model_id {str} -- the model id.
            outputRootPath {str} -- the data output root path.

        Keyword Arguments:
            model_type {str} -- two topic model types: 'default', 'mallet' (default: {'default'}).
            model_path {str} -- the path to 'mallet' model (default: {None}).
        """

        self.model_id = model_id
        self.outputRootPath = Path(outputRootPath)
        self.model_type = model_type
        self.model_path = model_path
        self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.id2word = None
        self.corpus = None
        self.corpus_bow = None
        self.optimal_model = None
        self.bigram_model = None
        self.trigram_model = None

    def runModel(self):
        pass

    def _preprocess_data(self, docs):
        """Preprocess Data by using methods in utility module.

        Arguments:
            docs {list} -- a list of raw docs texts.

        Returns:
            list -- a list of processed docs in list of tokens form.
        """

        docs = utility.remove_email(docs)
        docs = utility.remove_newline(docs)
        docs = utility.remove_single_quote(docs)
        docs_words = list(utility.doc_to_words(docs))

        docs_words = utility.remove_stopwords(docs_words)
        docs_words = utility.lemmatized(
            self.nlp, docs_words, allowed_postags=['NOUN', 'ADJ', 'ADV'])
        docs_words = utility.remove_stopwords(docs_words)

        return docs_words

    def _prepare_data_for_training(self, docs):
        """Generate the prerequisite data for training topic model.

        Arguments:
            docs {list} -- a list of processed docs in lists of tokens form.
        """

        outputFolderPath = self.outputRootPath / (self.model_type + "_model")
        if not outputFolderPath.exists():
            outputFolderPath.mkdir()

        # docs_words = self._preprocess_data(docs)

        if (outputFolderPath / "bigram").exists() and (outputFolderPath / "trigram").exists():
            print("load n-gram...")
            bigram = gensim.models.phrases.Phraser.load(
                str(outputFolderPath / "bigram"))
            trigram = gensim.models.phrases.Phraser.load(
                str(outputFolderPath / "trigram"))
        else:
            bigram, trigram = utility.build_n_gram(docs_words)
            bigram.save(str(outputFolderPath / "bigram"))
            trigram.save(str(outputFolderPath / "trigram"))

        self.bigram_model = gensim.models.phrases.Phraser(bigram)
        self.trigram_model = gensim.models.phrases.Phraser(trigram)

        # docs_words_trigram = utility.make_trigrams(
        #     self.trigram_model, docs_words)

        if (outputFolderPath / 'dictionary').exists():
            print("load dictionary...")
            self.id2word = gensim.corpora.dictionary.Dictionary.load(
                str(outputFolderPath / 'dictionary'))
        else:
            self.id2word = corpora.Dictionary(docs_words_trigram)
            self.id2word.save(str(outputFolderPath / 'dictionary'))

        if (outputFolderPath / 'corpus.pkl').exists():
            print("load corpus.pkl...")
            with (outputFolderPath / 'corpus.pkl').open("rb") as fp:
                self.corpus = pickle.load(fp)
        else:
            self.corpus = docs_words_trigram
            with (outputFolderPath / 'corpus.pkl').open("wb") as fp:
                pickle.dump(self.corpus, fp)

        if (outputFolderPath / 'corpus_bow.pkl').exists():
            print("load corpus_bow.pkl...")
            with (outputFolderPath / 'corpus_bow.pkl').open("rb") as fp:
                self.corpus_bow = pickle.load(fp)
        else:
            self.corpus_bow = [self.id2word.doc2bow(
                doc) for doc in self.corpus]
            with (outputFolderPath / 'corpus_bow.pkl').open("wb") as fp:
                pickle.dump(self.corpus_bow, fp)

    def _compute_coherence_values(self, limit, start=2, step=2):
        """Compute the coherence values for topic model with different topic number.

        Arguments:
            limit {int} -- the upper bound of the topic number.

        Keyword Arguments:
            start {int} -- the initial topic number (default: {2}).
            step {int} -- the gap between two test topic number (default: {2}).

        Returns:
            tuple -- (a list of models, a list of coherences)
        """

        outputFolderPath = self.outputRootPath / (self.model_type + "_model")

        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            if self.model_type == "mallet":
                model = gensim.models.wrappers.LdaMallet(
                    self.model_path, corpus=self.corpus_bow,
                    id2word=self.id2word, num_topics=num_topics)
            else:
                model = gensim.models.ldamodel.LdaModel(
                    corpus=self.corpus_bow,
                    id2word=self.id2word, num_topics=num_topics
                )
            model_list.append(model)
            coherencemodel = CoherenceModel(
                model=model, texts=self.corpus, dictionary=self.id2word,
                coherence="c_v")
            coherence_values.append(coherencemodel.get_coherence())
            # print(coherence_values)

        self._plot_topic_num_to_coherence(
            coherence_values, limit, start, step)

        with (outputFolderPath / "coherence_values.json").open("w") as fp:
            json.dump(coherence_values, fp, indent=4)
        return model_list, coherence_values

    def _plot_topic_num_to_coherence(self, coherence_values, limit, start=2,
                                     step=2):
        """Plot the relationship between topic number and corresponding coherence.

        Arguments:
            coherence_values {list} -- a list of coherence
            limit {int} -- upper bound of topic number

        Keyword Arguments:
            start {int} -- initial topic number (default: {2}).
            step {int} -- gap between two test topic number (default: {2}).
        """

        outputFolderPath = self.outputRootPath / (self.model_type + "_model")
        if not outputFolderPath.exists():
            outputFolderPath.mkdir()
        outputFilePath = outputFolderPath / "topic_num_to_coherence_plot.png"

        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.savefig(str(outputFilePath))

    def train_model(self, docs, limit, start, step):
        """Train models and get one with highest coherence.

        Arguments:
            docs {list} -- a list of raw docs texts.
            limit {int} -- upper bound of topic number.
            start {int} -- initial number of topic number.
            step {int} -- gap between two test topic number.
        """

        outputFolderPath = self.outputRootPath / (self.model_type + "_model")
        outputFilePath = outputFolderPath / "optimal_model"

        print("Preparing data for training...")
        self._prepare_data_for_training(docs)

        if outputFilePath.exists():
            print("loaded model from {}".format(str(outputFilePath)))
            if self.model_type == "mallet":
                self.optimal_model = gensim.models.wrappers.ldamallet.LdaMallet.load(
                    str(outputFilePath))
                return
            if self.model_type == "default":
                self.optimal_model = gensim.models.ldamodel.LdaModel.load(
                    str(outputFilePath))
                return
        if not outputFolderPath.exists():
            outputFolderPath.mkdir()

        print("Start training...")
        model_list, coherence_values = self._compute_coherence_values(
            limit, start, step)

        max_coherence = 0
        best_topics_num = 0
        for index, (model, coherence_value) in enumerate(list(zip(model_list,
                                                                  coherence_values))):
            if coherence_value > max_coherence:
                max_coherence = coherence_value
                best_topics_num = start + index * step
                self.optimal_model = model

        self.optimal_model.save(str(outputFolderPath / "optimal_model"))
        print("The max coherence of the best model: ", max_coherence)
        print("The number of topics of the best model: ", best_topics_num)

    def _get_embedding(self, doc):
        embedding = self.optimal_model[doc]
        # print(embedding)
        return embedding

    def _get_item_profile(self, doc):
        item_profile = self._get_embedding(doc)

        return item_profile

    def get_items_profiles(self, docs):
        print(len(docs))
        items_contents_words = self._preprocess_data(docs)
        corpus_list = utility.make_trigrams(self.trigram_model, items_contents_words)
        corpus_bow_list = [self.id2word.doc2bow(corpus) for corpus in corpus_list]

        print("starting...")
        items_profiles_list = [self._get_item_profile(corpus_bow) for corpus_bow in corpus_bow_list]
        items_profiles = utility.transform_to_sparse_matrix(items_profiles_list)

        print("ending...")
        return items_profiles

    def load_items_profiles(self, items_ids, articles_df):
        corpus_indexes = []
        for item_id in items_ids:
            index = articles_df[articles_df['contentId']
                                == item_id].index.tolist()
            corpus_indexes += index
        
        docs_topics_distributions = list(self.optimal_model.load_document_topics())

        items_profiles_list = [docs_topics_distributions[corpus_index] for corpus_index in corpus_indexes]
        items_profiles = utility.transform_to_sparse_matrix(items_profiles_list)

        return items_profiles

    def get_user_profile(self, user_id, interactions_full_df, articles_df):
        """Get dataframe with whole information involving topics and docs.

        Arguments:
            user_id {int} -- the user id.
            articles_df {pandas.DataFrame} -- the article dataframe.
            interactions_full_df {pandas.DataFrame} -- the interactions dataframe.

        Returns:
            pandas.DataFrame -- the dataframe with while information involving
                                topics and docs.
        """
        outputFolderPath = self.outputRootPath / (self.model_type + "_model") / "information"
        if not outputFolderPath.exists():
            outputFolderPath.mkdir()
        
        items_ids, items_contents = utility.get_items(user_id, interactions_full_df, articles_df)
        # contentIds = interactions_full_df[interactions_full_df['personId']
        #                                   == user_id]['contentId'].tolist()

        items_profiles = self.load_items_profiles(items_ids, articles_df)
        print(items_profiles.shape)
        user_items_strengths = utility.get_weight(user_id, interactions_full_df, items_ids)
        # print(user_items_strengths)
        user_items_strengths = np.array(user_items_strengths).reshape(-1, 1)
        print(user_items_strengths.shape)
        user_items_strengths_weighted_avg = np.sum(items_profiles.multiply(
            user_items_strengths), axis=0) / np.sum(user_items_strengths)
        user_profile_strength_norm = sklearn.preprocessing.normalize(
            user_items_strengths_weighted_avg)
        with (outputFolderPath / (str(user_id) + "_strength_norm.pkl")).open("wb") as fp:
            pickle.dump(user_profile_strength_norm, fp)

        return user_profile_strength_norm

  

    # def _get_topics_to_cnt_norm(self, items_profiles):
    #     """Get weight based on the number of docs related to each topic.
        
    #     Arguments:
    #         items_profiles {pandas.DataFrame} -- the dataframe with information.
        
    #     Returns:
    #         dict -- topic_id -> number of related docs.
    #     """

    #     topics_to_cnt = items_profiles['Dominant_Topic'].value_counts(
    #         normalize=True).to_dict()

    #     topics_to_cnt = dict([(int(key), value)
    #                           for key, value in topics_to_cnt.items()])

    #     return topics_to_cnt

    # def _get_topics_to_strength_norm(self, items_profiles):
    #     """Get weight based on the strength of docs related to each topic.
        
    #     Arguments:
    #         items_profiles {pandas.DataFrame} -- the dataframe with information.
        
    #     Returns:
    #         dict -- topic_id -> strength
    #     """

    #     topics_to_strength = items_profiles.groupby(
    #         ['Dominant_Topic'])['Strength'].sum().to_dict()

    #     total = sum(topics_to_strength.values())
    #     topics_to_strength = dict([(int(key), value / total)
    #                                for key, value in topics_to_strength.items()])

    #     return topics_to_strength

    # def get_user_profile(self, user_id, articles_df, interactions_full_df):
    #     """Generate profile for user_id.
        
    #     Arguments:
    #         user_id {int} -- the user id.
    #         articles_df {pandas.DataFrame} -- the articles dataframe.
    #         interactions_full_df {pandas.DataFrame} -- the interactions dataframe.
    #     """

    #     outputFolderPath = self.outputRootPath / \
    #         (self.model_type + "_model") / "profiles"
    #     if not outputFolderPath.exists():
    #         outputFolderPath.mkdir()

    #     items_profiles = self.get_items_profiles(
    #         user_id, articles_df, interactions_full_df)

    #     topics_to_cnt_norm = self._get_topics_to_cnt_norm(
    #         items_profiles)
    #     with (outputFolderPath / (str(user_id) + "_cnt_norm.pkl")).open("wb") as fp:
    #         pickle.dump(topics_to_cnt_norm, fp)

    #     topics_to_strength_norm = self._get_topics_to_strength_norm(
    #         items_profiles)
    #     with (outputFolderPath / (str(user_id) + "_strength_norm.pkl")).open("wb") as fp:
    #         pickle.dump(topics_to_strength_norm, fp)

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
        """Get the socore for new incoming docs based on profile.
        
        Arguments:
            user_id {int} -- the user id.
            docs {list} -- a list of new incoming raw docs texts,
                           [[id, content], ...].
        
        Returns:
            tuple -- (doc_id -> score weighted by number of docs,
                      doc_id -> score weighted by strength)
        """

        user_profile_strength_norm = self.load_user_profile(user_id)

        items_ids, items_contents = zip(*docs)

        items_profiles = self.get_items_profiles(items_contents)

        cosine_similarities = cosine_similarity(
            user_profile_strength_norm, items_profiles)

        item_id_to_strength_weight_score = [(items_ids[i], cosine_similarities[0, i])
                                            for i in range(cosine_similarities.shape[1])]

        return item_id_to_strength_weight_score

    def recommend_items(self, user_id, docs, articles_df, items_to_ignore=[],
                        topn=10, verbose=False):
        """Recommend new incoming items to user_id.
        
        Arguments:
            user_id {int} -- the user id.
            docs {list} -- a list of new incoming raw text.
            articles_df {pandas.DataFrame} -- the articles dataframe to get additional information.
        
        Keyword Arguments:
            items_to_ignore {list} -- a list of doc_ids that user_id have been viewed (default: {[]}).
            topn {int} -- the number of recommended items (default: {10}).
            verbose {bool} -- the flag that indicates whether add additional information (default: {False}).
        
        Raises:
            Exception -- articles_df is not provided when verbose is True.

        Returns:
            pandas.Dataframe -- the dataframe of recommended items with information.
        """

        item_id_to_strength_weight_score = self.get_score_of_docs(
                user_id, docs)

        item_id_to_strength_weight_score.sort(key=lambda x: x[1], reverse=True)
        similar_items_filter = list(
            filter(lambda x: x[0] not in items_to_ignore, item_id_to_strength_weight_score))
        # filter out the items that are already interacted
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
