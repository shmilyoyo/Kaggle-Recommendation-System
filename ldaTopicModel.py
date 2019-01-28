from BaseModel import BaseModel
import preprocessData

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

    def __init__(self, model_id, outputRootPath, model_type='default', model_path=None):
        self.model_id = model_id
        self.outputRootPath = Path(outputRootPath)
        self.model_type = model_type
        self.model_path = model_path
        self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.id2word = None
        self.corpus = None
        self.corpus_bow = None
        self.optimal_model = None

    def runModel(self):
        pass

    def preprocess_data(self, docs):
        outputFolderPath = self.outputRootPath / (self.model_type + "_model")
        # if not outputFolderPath.exists():
        #     outputFolderPath.mkdir()

        # docs = preprocessData.remove_email(docs)
        # docs = preprocessData.remove_newline(docs)
        # docs = preprocessData.remove_single_quote(docs)
        # docs_words = list(preprocessData.doc_to_words(docs))

        # docs_words = preprocessData.remove_stopwords(docs_words)
        # docs_words = preprocessData.lemmatized(
        #     self.nlp, docs_words, allowed_postags=['NOUN', 'ADJ', 'ADV'])
        # docs_words = preprocessData.remove_stopwords(docs_words)

        # if (outputFolderPath / "bigram").exists() and (outputFolderPath / "trigram").exists():
        #     print("load n-gram...")
        #     bigram = gensim.models.phrases.Phraser.load(str(outputFolderPath / "bigram"))
        #     trigram = gensim.models.phrases.Phraser.load(str(outputFolderPath / "trigram"))
        # else:
        #     bigram, trigram = preprocessData.build_n_gram(docs_words)
        #     bigram.save(str(outputFolderPath / "bigram"))
        #     trigram.save(str(outputFolderPath / "trigram"))

        # bigram_model = gensim.models.phrases.Phraser(bigram)
        # trigram_model = gensim.models.phrases.Phraser(trigram)

        # docs_words_trigram = preprocessData.make_trigrams(
        #     trigram_model, docs_words)

        if (outputFolderPath / 'dictionary').exists():
            self.id2word = gensim.corpora.dictionary.Dictionary.load(
                str(outputFolderPath / 'dictionary'))
        else:
            self.id2word = corpora.Dictionary(docs_words_trigram)
            self.id2word.save(str(outputFolderPath / 'dictionary'))

        if (outputFolderPath / 'corpus.pkl').exists():
            with (outputFolderPath / 'corpus.pkl').open("rb") as fp:
                self.corpus = pickle.load(fp)
        else:
            self.corpus = docs_words_trigram
            with (outputFolderPath / 'corpus.pkl').open("wb") as fp:
                pickle.dump(self.corpus, fp)

        if (outputFolderPath / 'corpus_bow.pkl').exists():
            with (outputFolderPath / 'corpus_bow.pkl').open("rb") as fp:
                self.corpus_bow = pickle.load(fp)
        else:
            self.corpus_bow = [self.id2word.doc2bow(
                doc) for doc in self.corpus]
            with (outputFolderPath / 'corpus_bow.pkl').open("wb") as fp:
                pickle.dump(self.corpus_bow, fp)

    def compute_coherence_values(self, limit, start=2, step=2):
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
            print(coherence_values)

        self.get_topic_num_to_coherence_plot(
            coherence_values, limit, start, step)

        with (outputFolderPath / "coherence_values.json").open("w") as fp:
            json.dump(coherence_values, fp, indent=4)
        return model_list, coherence_values

    def get_topic_num_to_coherence_plot(self, coherence_values, limit, start=2, step=2):
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

    def get_best_model(self, limit, start, step):
        outputFolderPath = self.outputRootPath / (self.model_type + "_model")
        outputFilePath = outputFolderPath / "optimal_model"

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

        model_list, coherence_values = self.compute_coherence_values(
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

    def get_topics_docs_full_df_for_user(self, user_id, articles_df, interactions_full_df):
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
        user_contents_strength_df = interactions_full_df[interactions_full_df['personId'] == user_id]

        topics_docs_full_df_for_user = pd.DataFrame()

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
                    topics_docs_full_df_for_user = topics_docs_full_df_for_user.append(pd.Series(
                        [int(topic_id), round(prop_topic, 4), topic_keywords]),
                        ignore_index=True)
                else:
                    break
        topics_docs_full_df_for_user.columns = [
            'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Append the original text to the data frame
        contents = pd.Series(corpus)
        contentIds = pd.Series(contentIds)
        topics_docs_full_df_for_user = pd.concat(
            [topics_docs_full_df_for_user, contentIds, contents], axis=1)
        topics_docs_full_df_for_user = topics_docs_full_df_for_user.reset_index()
        topics_docs_full_df_for_user.columns = [
            'Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords',
            'contentId', 'Text']

        topics_docs_full_df_for_user = topics_docs_full_df_for_user.merge(
            user_contents_strength_df[[
                "contentId", "eventStrength"]], how="left",
            left_on="contentId", right_on="contentId")
        topics_docs_full_df_for_user.columns = [
            'Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords',
            'Content_Id', 'Text', "Strength"]
        return topics_docs_full_df_for_user

    # def get_topics_to_strength_for_doc_ids(self, topics_docs_full_df, doc_ids):
    #     topic_id_to_average_strength = topics_docs_full_df.groupby('Dominant_Topic')[
    #         'eventStrength'].mean().reset_index().set_index('Dominant_Topic'
    #                                                         ).to_dict()['eventStrength']
    #     topic_id_to_average_strength = dict(
    #         [(int(key), value) for key, value in topic_id_to_average_strength.items()])
    #     return topic_id_to_average_strength

    # def get_whole_words_distributions_for_topics(self):
    #     topic_id_to_whole_words_distributions = dict(
    #         enumerate(self.optimal_model.get_topics()))

    #     return topic_id_to_whole_words_distributions

    def get_topics_to_cnt(self, topics_docs_full_df_for_user):
        topics_to_cnt = topics_docs_full_df_for_user['Dominant_Topic'].value_counts(
        ).to_dict()

        topics_to_cnt = dict([(int(key), value)
                              for key, value in topics_to_cnt.items()])

        return topics_to_cnt

    def get_topics_to_strength(self, topics_docs_full_df_for_user):
        topics_to_strength = topics_docs_full_df_for_user.groupby(
            ['Dominant_Topic'])['eventStrength'].sum().to_dict()

        topics_to_strength = dict([(int(key), value)
                                   for key, value in topics_to_strength.items()])

        return topics_to_strength
