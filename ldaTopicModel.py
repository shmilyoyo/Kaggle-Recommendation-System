from BaseModel import BaseModel
import preprocessData

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
from gensim.models import CoherenceModel
import gensim.corpora as corpora

import matplotlib.pyplot as plt
# spacy for lemmatization
import spacy


class LdaTopicModel(BaseModel):

    def __init__(self, model_id):
        self.model_id = model_id
        self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.id2word = None
        self.corpus = None
        self.corpus_bow = None
        self.optimal_model = None

    def runModel(self):
        pass

    def preprocess_data(self, docs):
        docs = preprocessData.remove_email(docs)
        docs = preprocessData.remove_newline(docs)
        docs = preprocessData.remove_single_quote(docs)
        docs_words = list(preprocessData.doc_to_words(docs))

        docs_words = preprocessData.remove_stopwords(docs_words)
        docs_words = preprocessData.lemmatized(
            self.nlp, docs_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        docs_words = preprocessData.remove_stopwords(docs_words)
        
        bigram_model, trigram_model = preprocessData.build_n_gram(docs_words)

        docs_words_trigram = preprocessData.make_trigrams(
            trigram_model, docs_words)

        self.id2word = corpora.Dictionary(docs_words_trigram)
        self.corpus = docs_words_trigram
        self.corpus_bow = [self.id2word.doc2bow(doc) for doc in self.corpus]

    def compute_coherence_values(self, limit, start=2, step=2):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            print(num_topics)
            model = gensim.models.ldamodel.LdaModel(
                corpus=self.corpus_bow, id2word=self.id2word, num_topics=num_topics)
            model_list.append(model)
            coherencemodel = CoherenceModel(
                model=model, texts=self.corpus, dictionary=self.id2word,
                coherence="c_v")
            coherence_values.append(coherencemodel.get_coherence())

        self.get_topic_num_to_coherence_plot(coherence_values, limit, start, step)

        return model_list, coherence_values

    def get_topic_num_to_coherence_plot(self, coherence_values, limit, start=2, step=2):
        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.savefig('./topic_num_to_coherence_plot.png')

    def get_best_model(self, limit, start, step):
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
        
        self.optimal_model.to_pickle("./optimal_model.pkl")
        
        print("The max coherence of the best model: ", max_coherence)
        print("The number of topics of the best model: ", best_topics_num)

    def get_topics_docs_full_df(self, corpus_bow, docIds, contents_strength_df):
        topics_docs_full_df = pd.DataFrame()

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
                    topics_docs_full_df = topics_docs_full_df.append(pd.Series(
                        [int(topic_id), round(prop_topic, 4), topic_keywords]),
                        ignore_index=True)
                else:
                    break
        topics_docs_full_df.columns = [
            'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Append the original text to the data frame
        contents = pd.Series(self.corpus)
        docIds = pd.Series(docIds)
        topics_docs_full_df = pd.concat(
            [topics_docs_full_df, docIds, contents], axis=1)
        topics_docs_full_df = topics_docs_full_df.reset_index()
        topics_docs_full_df.columns = [
            'Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords',
            'Documnet_Id', 'Text']

        topics_docs_full_df = topics_docs_full_df.merge(
            contents_strength_df, how="left", left_on="Documnet_Id",
            right_on="contentId")

        return topics_docs_full_df

    def get_topic_id_to_average_strength(self, topics_docs_full_df):
        topic_id_to_average_strength = topics_docs_full_df.groupby('Dominant_Topic')[
            'eventStrength'].mean().reset_index().set_index('Dominant_Topic'
                                                            ).to_dict()['eventStrength']

        return topic_id_to_average_strength
    
    def get_whole_words_distributions_for_topics(self, topic_ids):
        topic_id_to_whole_words_distributions = {}
        whole_words_distributions = self.optimal_model.get_topics()
        for topic_id in topic_ids:
            topic_id_to_whole_words_distributions[topic_id] = whole_words_distributions[topic_ids]
        
        return topic_id_to_whole_words_distributions