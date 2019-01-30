import errno
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

import gensim
import gensim.corpora as corpora
import utility
from content_based_model import ContentedBasedModel
from gensim.models import CoherenceModel


class LdaTopicModel(ContentedBasedModel):
    """
    Recommendation model based on LDA topic model.

    Arguments:
        ContentedBasedModel {class} -- the base class.
    """

    def __init__(self, model_id, output_root_path, model_path=None):
        """
        Initialize the parameters.

        Arguments:
            model_id {str} -- the model id.
            output_root_path {str} -- the data output root path.

        Keyword Arguments:
            model_path {str} -- the path to 'mallet' model (default: {None}).
        """
        super().__init__(model_id, output_root_path)
        self.model_path = model_path
        self.id2word = None
        self.corpus = None
        self.corpus_bow = None
        self.bigram_model = None
        self.trigram_model = None

    def get_model_name(self):
        """
        Get the model name.
        
        Returns:
            str -- the model id.
        """

        return self.model_id

    def _prepare_data_for_training(self, docs):
        """
        Generate the prerequisite data for training topic model.

        Arguments:
            docs {list} -- a list of processed docs in lists of tokens form.
        """

        outputFolderPath = self.output_root_path / (self.model_id + "_model")
        if not outputFolderPath.exists():
            outputFolderPath.mkdir()

        docs_words = super().preprocess_data(docs)

        bigram, trigram = utility.build_n_gram(docs_words)
        bigram.save(str(outputFolderPath / "bigram"))
        trigram.save(str(outputFolderPath / "trigram"))

        self.bigram_model = gensim.models.phrases.Phraser(bigram)
        self.trigram_model = gensim.models.phrases.Phraser(trigram)

        docs_words_trigram = utility.make_trigrams(
            self.trigram_model, docs_words)

        self.id2word = corpora.Dictionary(docs_words_trigram)
        self.id2word.save(str(outputFolderPath / 'dictionary'))

        self.corpus = docs_words_trigram
        with (outputFolderPath / 'corpus.pkl').open("wb") as fp:
            pickle.dump(self.corpus, fp)

        self.corpus_bow = [self.id2word.doc2bow(
            doc) for doc in self.corpus]
        with (outputFolderPath / 'corpus_bow.pkl').open("wb") as fp:
            pickle.dump(self.corpus_bow, fp)

    def _compute_coherence_values(self, limit, start=2, step=2):
        """
        Compute the coherence values for topic model with different topic number.

        Arguments:
            limit {int} -- the upper bound of the topic number.

        Keyword Arguments:
            start {int} -- the initial topic number (default: {2}).
            step {int} -- the gap between two test topic number (default: {2}).

        Returns:
            tuple -- (a list of models, a list of coherences)
        """

        outputFolderPath = self.output_root_path / (self.model_id + "_model")

        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            if self.model_id == "mallet":
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

        self._plot_topic_num_to_coherence(
            coherence_values, limit, start, step)

        with (outputFolderPath / "coherence_values.json").open("w") as fp:
            json.dump(coherence_values, fp, indent=4)
        return model_list, coherence_values

    def _plot_topic_num_to_coherence(self, coherence_values, limit, start=2,
                                     step=2):
        """
        Plot the relationship between topic number and corresponding coherence.

        Arguments:
            coherence_values {list} -- a list of coherence
            limit {int} -- upper bound of topic number

        Keyword Arguments:
            start {int} -- initial topic number (default: {2}).
            step {int} -- gap between two test topic number (default: {2}).
        """

        outputFolderPath = self.output_root_path / (self.model_id + "_model")
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
        """
        Train models and get one with highest coherence.

        Arguments:
            docs {list} -- a list of raw docs texts.
            limit {int} -- upper bound of topic number.
            start {int} -- initial number of topic number.
            step {int} -- gap between two test topic number.
        """

        outputFolderPath = self.output_root_path / (self.model_id + "_model")

        print("Preparing data for training...")
        self._prepare_data_for_training(docs)

        if not outputFolderPath.exists():
            outputFolderPath.mkdir()

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

        items_profiles = self.optimal_model.load_document_topics()
        with (outputFolderPath / "items_profiles.pkl").open("wb") as fp:
            pickle.dump(items_profiles, fp)

        print("The max coherence of the best model: ", max_coherence)
        print("The number of topics of the best model: ", best_topics_num)

    def load_model(self):
        """
        Load the model.
        
        Raises:
            FileNotFoundError -- raise error when model is not in path.
            FileNotFoundError -- raise error when n-gram is not in the path.
            FileNotFoundError -- raise error when dictionary is not in the path.
            FileNotFoundError -- raise error when corpus is not in the path.
            FileNotFoundError -- raise error when corpus_bow is not in the path.
        """

        outputFolderPath = self.output_root_path / (self.model_id + "_model")

        if (outputFolderPath / "optimal_model").exists():
            print("loaded model from {}".format(str(outputFolderPath / "optimal_model")))
            if self.model_id == "mallet":
                self.optimal_model = gensim.models.wrappers.ldamallet.LdaMallet.load(
                    str(outputFolderPath / "optimal_model"))
            if self.model_id == "default":
                self.optimal_model = gensim.models.ldamodel.LdaModel.load(
                    str(outputFolderPath / "optimal_model"))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(outputFolderPath / "optimal_model"))
        
        if (outputFolderPath / "bigram").exists() and (outputFolderPath / "trigram").exists():
            print("load n-gram...")
            bigram = gensim.models.phrases.Phraser.load(
                str(outputFolderPath / "bigram"))
            trigram = gensim.models.phrases.Phraser.load(
                str(outputFolderPath / "trigram"))

            self.bigram_model = gensim.models.phrases.Phraser(bigram)
            self.trigram_model = gensim.models.phrases.Phraser(trigram)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(outputFolderPath / "bigram"))
            
        
        if (outputFolderPath / 'dictionary').exists():
            print("load dictionary...")
            self.id2word = gensim.corpora.dictionary.Dictionary.load(
                str(outputFolderPath / 'dictionary'))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(outputFolderPath / 'dictionary'))
            
        
        if (outputFolderPath / 'corpus.pkl').exists():
            print("load corpus.pkl...")
            with (outputFolderPath / 'corpus.pkl').open("rb") as fp:
                self.corpus = pickle.load(fp)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(outputFolderPath / 'corpus.pkl'))
            
        
        if (outputFolderPath / 'corpus_bow.pkl').exists():
            print("load corpus_bow.pkl...")
            with (outputFolderPath / 'corpus_bow.pkl').open("rb") as fp:
                self.corpus_bow = pickle.load(fp)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(outputFolderPath / 'corpus_bow.pkl'))

    def get_embedding(self, doc):
        """
        Get embedding representation for doc text.
        
        Arguments:
            doc {str} -- the doc text.
        
        Returns:
            matrix -- the embedding of the doc text.
        """

        embedding = self.optimal_model[doc]

        return embedding

    def get_items_profiles(self, docs):
        """
        Get a list of vectors corresponding to docs in matrix.

        Arguments:
            docs {list} -- a list of doc text.

        Returns:
            sparse matrix -- the total vectors corresponding to docs.
        """

        items_contents_words = self.preprocess_data(docs)
        corpus_list = utility.make_trigrams(self.trigram_model, items_contents_words)
        corpus_bow_list = [self.id2word.doc2bow(corpus) for corpus in corpus_list]

        items_profiles_list = [self.get_item_profile(corpus_bow) for corpus_bow in corpus_bow_list]
        items_profiles = utility.transform_tuple_to_sparse_matrix(items_profiles_list)

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
        items_profiles = utility.transform_tuple_to_sparse_matrix(items_profiles_list)

        return items_profiles
