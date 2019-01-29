from base_model import BaseModel
import preprocess_data

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
        """Preprocess Data by using methods in preprocess_data module.

        Arguments:
            docs {list} -- a list of raw docs texts.

        Returns:
            list -- a list of processed docs in list of tokens form.
        """

        docs = preprocess_data.remove_email(docs)
        docs = preprocess_data.remove_newline(docs)
        docs = preprocess_data.remove_single_quote(docs)
        docs_words = list(preprocess_data.doc_to_words(docs))

        docs_words = preprocess_data.remove_stopwords(docs_words)
        docs_words = preprocess_data.lemmatized(
            self.nlp, docs_words, allowed_postags=['NOUN', 'ADJ', 'ADV'])
        docs_words = preprocess_data.remove_stopwords(docs_words)

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
            bigram, trigram = preprocess_data.build_n_gram(docs_words)
            bigram.save(str(outputFolderPath / "bigram"))
            trigram.save(str(outputFolderPath / "trigram"))

        self.bigram_model = gensim.models.phrases.Phraser(bigram)
        self.trigram_model = gensim.models.phrases.Phraser(trigram)

        # docs_words_trigram = preprocess_data.make_trigrams(
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
            print("model existed in {}".format(str(outputFilePath)))
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

    def get_items_profiles(self, user_id, articles_df,
                                          interactions_full_df):
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
        
        contentIds = interactions_full_df[interactions_full_df['personId']
                                          == user_id]['contentId'].tolist()
        corpus_indexes = []
        for contentId in contentIds:
            index = articles_df[articles_df['contentId']
                                == contentId].index.tolist()
            corpus_indexes += index

        corpus = [self.corpus[corpus_index] for corpus_index in corpus_indexes]
        corpus_bow = [self.corpus_bow[corpus_index]
                      for corpus_index in corpus_indexes]

        # test
        # doc = articles_df.loc[corpus_indexes[0]].text
        # cp = self.corpus[corpus_indexes[0]]
        # print("test")
        # print(doc)
        # print(cp)

        user_contents_strength_df = interactions_full_df[
            interactions_full_df['personId'] == user_id]

        items_profiles = pd.DataFrame()

        # find one dominant topic for each document
        for row in self.optimal_model[corpus_bow]:
            row = sorted(row, key=lambda x: x[1], reverse=True)
            # get topic number, confidence, and keywords
            for j, (topic_id, prop_topic) in enumerate(row):
                if j == 0:
                    word_distribution = self.optimal_model.show_topic(
                        topic_id)
                    topic_keywords = ", ".join(
                        [word for word, prop in word_distribution])
                    items_profiles = items_profiles.append(pd.Series(
                        [int(topic_id), round(prop_topic, 4), topic_keywords]),
                        ignore_index=True)
                else:
                    break
        items_profiles.columns = [
            'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Append the original text to the data frame
        contents = pd.Series(corpus)
        contentIds = pd.Series(contentIds)
        items_profiles = pd.concat(
            [items_profiles, contentIds, contents], axis=1)
        items_profiles = items_profiles.reset_index()
        items_profiles.columns = [
            'Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords',
            'contentId', 'Text']

        items_profiles = items_profiles.merge(
            user_contents_strength_df[[
                "contentId", "eventStrength"]], how="left",
            left_on="contentId", right_on="contentId")
        items_profiles.columns = [
            'Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords',
            'Content_Id', 'Text', "Strength"]
        
        outputFilePath = outputFolderPath / str(user_id) + ".pkl"
        items_profiles.to_pickle(str(outputFilePath))

        return items_profiles

    def _get_topics_to_cnt_norm(self, items_profiles):
        """Get weight based on the number of docs related to each topic.
        
        Arguments:
            items_profiles {pandas.DataFrame} -- the dataframe with information.
        
        Returns:
            dict -- topic_id -> number of related docs.
        """

        topics_to_cnt = items_profiles['Dominant_Topic'].value_counts(
            normalize=True).to_dict()

        topics_to_cnt = dict([(int(key), value)
                              for key, value in topics_to_cnt.items()])

        return topics_to_cnt

    def _get_topics_to_strength_norm(self, items_profiles):
        """Get weight based on the strength of docs related to each topic.
        
        Arguments:
            items_profiles {pandas.DataFrame} -- the dataframe with information.
        
        Returns:
            dict -- topic_id -> strength
        """

        topics_to_strength = items_profiles.groupby(
            ['Dominant_Topic'])['Strength'].sum().to_dict()

        total = sum(topics_to_strength.values())
        topics_to_strength = dict([(int(key), value / total)
                                   for key, value in topics_to_strength.items()])

        return topics_to_strength

    def get_user_profile(self, user_id, articles_df, interactions_full_df):
        """Generate profile for user_id.
        
        Arguments:
            user_id {int} -- the user id.
            articles_df {pandas.DataFrame} -- the articles dataframe.
            interactions_full_df {pandas.DataFrame} -- the interactions dataframe.
        """

        outputFolderPath = self.outputRootPath / \
            (self.model_type + "_model") / "profiles"
        if not outputFolderPath.exists():
            outputFolderPath.mkdir()

        items_profiles = self.get_items_profiles(
            user_id, articles_df, interactions_full_df)

        topics_to_cnt_norm = self._get_topics_to_cnt_norm(
            items_profiles)
        with (outputFolderPath / (str(user_id) + "_cnt_norm.pkl")).open("wb") as fp:
            pickle.dump(topics_to_cnt_norm, fp)

        topics_to_strength_norm = self._get_topics_to_strength_norm(
            items_profiles)
        with (outputFolderPath / (str(user_id) + "_strength_norm.pkl")).open("wb") as fp:
            pickle.dump(topics_to_strength_norm, fp)

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
        with (outputFolderPath / (str(user_id) + "_cnt_norm.pkl")).open("rb") as fp:
            topics_to_cnt_norm = pickle.load(fp)
        with (outputFolderPath / (str(user_id) + "_strength_norm.pkl")).open("rb") as fp:
            topics_to_strength_norm = pickle.load(fp)

        return topics_to_cnt_norm, topics_to_strength_norm

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

        doc_ids, docs = zip(*docs)

        user_profile_cnt_norm, user_profile_strength_norm = self.load_user_profile(
            user_id)
        topics = set(list(user_profile_cnt_norm))

        docs_words = self._preprocess_data(docs)
        corpus = preprocess_data.make_trigrams(self.trigram_model, docs_words)
        corpus_bow = [self.id2word.doc2bow(doc) for doc in corpus]

        doc_id_to_cnt_weight_score = {}
        doc_id_to_strength_weight_score = {}
        for doc_id, row in zip(doc_ids, self.optimal_model[corpus_bow]):
            row = list(filter(lambda x: x[0] in topics, row))
            # get topic number, confidence, and keywords
            cnt_weight_score = 0
            strength_weight_score = 0
            for topic_id, prop_topic in row:
                cnt_weight_score += user_profile_cnt_norm[topic_id] * prop_topic
                strength_weight_score += user_profile_strength_norm[topic_id] * prop_topic
            doc_id_to_cnt_weight_score[doc_id] = cnt_weight_score
            doc_id_to_strength_weight_score[doc_id] = strength_weight_score

        return doc_id_to_cnt_weight_score, doc_id_to_strength_weight_score

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

        doc_id_to_cnt_weight_score, \
            doc_id_to_strength_weight_score = self.get_score_of_docs(
                user_id, docs)

        doc_id_to_cnt_weight_score = sorted(
            doc_id_to_cnt_weight_score.items(), key=lambda x: x[1], reverse=True)
        doc_id_to_strength_weight_score = sorted(
            doc_id_to_strength_weight_score.items(), key=lambda x: x[1], reverse=True)
        # filter out the items that are already interacted
        similar_items_filter = list(
            filter(lambda x: x[0] not in items_to_ignore, doc_id_to_cnt_weight_score))

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
