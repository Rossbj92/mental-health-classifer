#Data manipulation
import numpy as np
import pandas as pd

#Feature engineering
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

#Text processing
from collections import Counter
import spacy

#Saving
from joblib import dump
#Grabbing num of available cores
import multiprocessing

nlp = spacy.load("en_core_web_sm")

def load_process_text(file):
    """Loads CSV and processes text data.

    This function tokenizes and lemmatizes text, as well as
    returning total text length. The CSV should contain 'title'
    and 'text' columns that are created in the 'cleaning_util.py' file.

    Args:
        file (csv): A CSV containing 'title' and 'text' columns.

    Returns:
        A Pandas dataframe with 4 new columns: 'total_text'
        (raw text), 'tokens' (tokenized text), 'lemmatized'
        (lemmatized text), and 'post_length' (total post length).
    """
    df = pd.read_csv(file)

    assert 'title' in df.columns, 'No "title" column found.'
    assert 'text' in df.columns, 'No "text" column found.'

    df['total_text'] = df['title'] + '. ' + df['text']
    df['tokens'] = df['total_text'].map(lambda x: nlp(x.lower()
                                                       .replace('\n','')
                                                       .strip()))
    df['lemmatized'] = df['tokens'].map(lambda x: [tok.lemma_ for tok in x if tok.is_punct == False and tok.is_stop == False and str(tok) not in '                 '])
    df['post_length'] = df['tokens'].map(lambda x: len(x))
    return df

def class_word_counts(df, class_):
    """Calculates word counts for a specified group.

    Args:
        class_ (str): A string that corresponds to a grouping
          variable to be filtered in df.

    Returns:
        A collections.Counter object containing word counts
        for each class.

    """
    counts = Counter()
    doc_lst = df.loc[df['sub'] == class_, 'lemmatized'].tolist()
    for doc in doc_lst:
        for token in doc:
            counts[token] += 1
    return counts

def class_count_dict(df,
                     classes,
                     top_words = 10
                     ):
    """Constructs a dictionary of class keys and word count values.

    Args:
        classes (list): A list that contains grouping variables to be
          filtered in df.
        top_words (int): An integer (default = 10) indicating the amount of
          words to be returned; words are in descending frequency (e.g.,
          1 = the most common word for that class).

    Returns:
        A dictionary with a key for each item in 'classes' and a
        collections.Counter object containing word count values

    """
    class_counts = {}
    for class_ in ['depression', 'anxiety', 'adhd', 'non_clinical']:
        class_counts[class_] = class_word_counts(df, class_).most_common(top_words)

    return class_counts

class D2V:
    """Tags documents and fits a Gensim Doc2Vec model.

    Using a Pandas dataframe, this can be used to fit a
    Doc2Vec model to train custom document embeddings.
    Default parameters for the model are for a PV-DM model,
    and PV-DBOW embeddings can also be computed. Additionally,
    Doc2Vec parameters can be manually modified.

    Attributes:
        train (obj): A Pandas dataframe containing the 'lemmatized'
          column from ___(cleaning method).
        model_type (str): Doc2Vec method to use. 'dm' by default, but
          also accepts 'dbow'.
        docs_tagged (obj): A list of TaggedDocuments, assigned through
          the 'tagged_docs' method.
    """
    def __init__(self, train, model_type = 'dm'):
        """Inits class with train, test, and model_type."""
        self.train = train
        self.model_type = model_type
        self.docs_tagged = None

    def tag_docs(self):
        """Return a list of tagged documents formatted for Doc2Vec.

        Method iterates through the 'lemmatized' train column, turning
        each into its own document with its index as the label. Must be
        called before the `model_train` method.

        Returns:
            self.
        """
        assert 'lemmatized' in self.train.columns, 'No "lemmatized" column found.'

        tagged_docs = []

        for idx, val in self.train['lemmatized'].iteritems():
            tagged_docs.append(TaggedDocument(self.train.loc[idx, 'lemmatized'], [idx]))

        self.docs_tagged = tagged_docs

        return self


    def model_train(self,
                    vector_size = 400,
                    dm = None,
                    dbow_words = None,
                    min_count = 10,
                    epochs = 40,
                    workers = multiprocessing.cpu_count()
                    ):
        """Trains a Doc2Vec model.

        This method is an aggregation of the several steps needed to train
        a Doc2Vec model. The model is first instantiated with a vocabulary
        based on the tagged documents, and the model is then trained on these
        data. For further documentation, see Gensim's official Doc2Vec docs
        (https://radimrehurek.com/gensim/models/doc2vec.html).

        Args:
            See https://radimrehurek.com/gensim/models/doc2vec.html.

        Returns:
            A trained Doc2Vec model.
        """
        assert self.docs_tagged != None, 'No TaggedDocuments found. Please run "tag_docs" method prior to "model_train".'

        if self.model_type == 'dbow':
            dm = 0
            dbow_words = 1
            model = Doc2Vec(vector_size = vector_size,
                            dm = dm,
                            dbow_words = dbow_words,
                            min_count = min_count,
                            epochs = epochs,
                            workers = workers
                           )
        else:
            model = Doc2Vec(vector_size = vector_size,
                            min_count = min_count,
                            epochs = epochs,
                            workers = workers
                           )
        model.build_vocab(self.docs_tagged)
        model.train(self.docs_tagged,
                    total_examples = model.corpus_count,
                    epochs = model.epochs
                   )

        return model


class GetVectors(D2V):
    """Used to obtain tf-idf scores and a variety of word vectors.

    Four different text features can be extracted using the methods below.
    `d2v_vecs` returns document vectors, `w2v_vecs` returns mean of a document's
    word embeddings, `tfidf_transform` returns tf-idf sparse matrices, and `tf_mowe`
    returns the mean of term frequency weighted word embeddings of a document.

    Attributes:
        train (obj): Training data in the form of a Pandas dataframe.
        test (obj): Test data in the form of a Pandas dataframe.
        model (obj): A trained Gensim Doc2Vec model.
    """
    def __init__(self, train, test, model):
        """Inits class with train, test, and model."""
        self.train = train
        self.test = test
        self.model = model
        self.dim = model.vector_size
        self.tf_fit = None
        self.words_weights = None
        self.tf_train_corpus = None

    def d2v_vecs(self):
        """Obtains document vectors.

        Returns:
            Numpy arrays of document vectors for train and test sets.
        """
        assert 'lemmatized' in self.train.columns and 'lemmatized' in self.test.columns, 'No "lemmatized" column found.'

        vecs_train = np.array(list(self.train['lemmatized'].map(lambda x: self.model.infer_vector(x))), dtype = 'float')
        vecs_test = np.array(list(self.test['lemmatized'].map(lambda x: self.model.infer_vector(x))), dtype = 'float')
        return vecs_train, vecs_test

    def w2v_vecs(self):
        """Obtains mean of document word embeddings.

        For each observation in the `lemmatized` series of train/test, each word in
        the observation is replaced with its vector from self.model. If the word is
        not in the model's vocabulary, a vector of 0s is returned. If no words in the
        observation are in the vocabulary, a vector of 0s is returned. The mean of all
        word embeddings in the observation are otherwrise calculated. Code adapted from
        http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/

        Returns:
            Numpy arrays of mean document word embeddings.
        """
        assert 'lemmatized' in self.train.columns and 'lemmatized' in self.test.columns, 'No "lemmatized" column found.'

        #
        mowe_train = (np.array([np.mean([self.model[word] for word in val if word in self.model.wv.vocab] or
                                        [np.zeros(self.dim)], axis = 0) for idx, val in self.train['lemmatized'].iteritems()]))
        mowe_test = (np.array([np.mean([self.model[word] for word in val if word in self.model.wv.vocab] or
                                       [np.zeros(self.dim)], axis = 0) for idx, val in self.test['lemmatized'].iteritems()]))
        return mowe_train, mowe_test

    def tfidf_fit(self):
        """Fit sci-kit learn TfidfVectorizer.

        A TfidfVectorizer is fit using the vocabulary from trained Gensim Doc2Vec
        model. Additionally, a dictionary mapping words to idf weights is created
        for use with the `tf_mowe` method.

        Returns:
            self.
        """
        assert 'lemmatized' in self.train.columns and 'lemmatized' in self.test.columns, 'No "lemmatized" column found.'
        train_corpus = self.train['lemmatized'].map(lambda x: ' '.join([word for word in x])).tolist()
        vectorizer = TfidfVectorizer(vocabulary = self.model.wv.vocab.keys())
        self.tf_fit = vectorizer.fit(train_corpus)
        self.tf_train_corpus = train_corpus
        self.words_weights = dict(zip(self.tf_fit.get_feature_names(), self.tf_fit.idf_))
        return self

    def tfidf_transform(self):
        """Extracts Tf-idf vectors for train and test data.

        Returns:
            Sparse tf-idf matrices.
        """
        assert 'lemmatized' in self.train.columns and 'lemmatized' in self.test.columns, 'No "lemmatized" column found.'
        test_corpus = self.test['lemmatized'].map(lambda x: ' '.join([word for word in x])).tolist()
        train_tfidf = self.tf_fit.transform(self.tf_train_corpus)
        test_tfidf = self.tf_fit.transform(test_corpus)
        return train_tfidf, test_tfidf

    def tf_mowe(self):
        """Extracts idf-weighted mean word embeddings.

        For each observation in the `lemmatized` series of train/test, each word in
        the observation is replaced with its vector from self.model and multiplied
        by its Idf weight. If the word is not in the model's vocabulary, a vector of
        0s is returned. If no words in the observation are in the vocabulary, a vector
        of 0s is returned. The mean of all word embeddings in the observation are
        otherwise calculated. Code adapted from:
        http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/

        Returns:
            Numpy arrays containing idf-weighted mean word vectors.
        """
        assert 'lemmatized' in self.train.columns and 'lemmatized' in self.test.columns, 'No "lemmatized" column found.'
        assert self.words_weights != 'None', 'No word_weights found. Run `tfidf_fit` method.'
        #weights = self.tf_fit.idf_
        idf_word_weights = dict(zip(self.tf_fit.get_feature_names(), self.words_weights))
        mowe_idf_train = (np.array([np.mean([self.model[word]*self.words_weights[word] for word in val if word in self.model.wv.vocab]
                                            or [np.zeros(self.dim)], axis = 0) for idx, val in self.train['lemmatized'].iteritems()]))
        mowe_idf_test = (np.array([np.mean([self.model[word]*self.words_weights[word] for word in val if word in self.model.wv.vocab]
                                            or [np.zeros(self.dim)], axis = 0) for idx, val in self.test['lemmatized'].iteritems()]))
        return mowe_idf_train, mowe_idf_test
