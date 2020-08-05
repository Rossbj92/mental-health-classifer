#Data manipulation
import numpy as np
import pandas as pd
#Text processing
import spacy
#Feature engineering
from sklearn.model_selection import train_test_split
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
#Saving
from joblib import dump
#Grabbing num of available cores
import multiprocessing

nlp = spacy.load("en_core_web_sm")

def load_process_text(file):
    """Short summary.
    
    Longer summary.
    
    Returns:
        df
    """
    df = pd.read_pickle(file, compression = 'bz2')
    df['total_text'] = df['title'] + '. ' + df['text']
    df['tokens'] = df['total_text'].map(lambda x: nlp(x.lower()
                                                      .replace('\n','')
                                                      .strip()))
    df['lemmatized'] = df['tokens'].map(lambda x: [tok.lemma_ for tok in x if tok.is_punct == False and tok.is_stop == False and str(tok) not in '                 '])
    df['post_length'] = df['tokens'].map(lambda x: len(x))
    return df

class D2V:
    """Tags documents and fits a Gensim Doc2Vec model.
    
    Using a Pandas dataframe, this can be used to fit a 
    Doc2Vec model to train custom document embeddings.
    Default parameters for the model are for a PV-DM model,
    and PV-DBOW embeddings can also be computed. Additionally,
    Doc2Vec parameters can be manually modified.
    
    Attributes:
        train: a Pandas dataframe containing the 'lemmatized' column from ___(cleaning method).
        model_type: Doc2Vec method to use. 'dm' by default, but also accepts 'dbow'.
        docs_tagged: a list of TaggedDocuments, assigned through the 'tagged_docs' method.
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
   
    def model_train(self, vector_size = 400, dm = None, dbow_words = None, min_count = 10, epochs = 40, workers = multiprocessing.cpu_count()):
        """Trains a Doc2Vec model.
        
        This method is an aggregation of the several steps needed to train
        a Doc2Vec model. The model is first instantiated with a vocabulary 
        based on the tagged documents, and the model is then trained on these
        data. For further documentation, see Gensim's official Doc2Vec docs. 
        
        Args:
            See https://radimrehurek.com/gensim/models/doc2vec.html.
        
        Returns:
            model: trained Doc2Vec model.
        """
        assert self.docs_tagged != None, 'No TaggedDocuments found. Please run "tag_docs" method prior to "model_train".'
        
        if self.model_type == 'dbow':
            dm = 0
            dbow_words = 1
        
        model = Doc2Vec(vector_size = vector_size, dm = dm, dbow_words = dbow_words, min_count = min_count, epochs = epochs, workers = workers)
        model.build_vocab(self.docs_tagged)
        model.train(self.docs_tagged, total_examples = model.corpus_count, epochs = model.epochs)
        
        return model
    

class GetVectors(D2V):
    """Used to obtain tf-idf scores and a variety of word vectors.
    
    Four different text features can be extracted using the methods below.
    `d2v_vecs` returns document vectors, `w2v_vecs` returns mean of a document's
    word embeddings, `tfidf_transform` returns tf-idf sparse matrices, and `tf_mowe`
    returns the mean of term frequency weighted word embeddings of a document. 
    
    Attributes:
        train: training data in the form of a Pandas dataframe.
        test: test data in the form of a Pandas dataframe.
        model: a trained Gensim Doc2Vec model. 
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
        
        Returns:
            Numpy arrays of mean document word embeddings.
        """
        assert 'lemmatized' in self.train.columns and 'lemmatized' in self.test.columns, 'No "lemmatized" column found.'
        
        #Code adapted from http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
        mowe_train = (np.array([np.mean([self.model[word] for word in val if word in self.model.wv.vocab] or 
                                        [np.zeros(self.dim)], axis = 0) for idx, val in self.train['lemmatized'].iteritems()]))
        mowe_test = (np.array([np.mean([self.model[word] for word in val if word in self.model.wv.vocab] or 
                                       [np.zeros(self.dim)], axis = 0) for idx, val in self.test['lemmatized'].iteritems()]))
        return mowe_train, mowe_test

    def tfidf_fit(self):
        """Fit sci-kit learn TfidfVectorizer.
         
        A TfidfVectorizer is fit using the vocabulary from trained Gensim Doc2Vec
        model. Additionally, a dictionary mapping words to idf weights is created
        to be used with the `tf_mowe` method.
        
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
        """Short summary.
        
        Returns:
        
        """
        assert 'lemmatized' in self.train.columns and 'lemmatized' in self.test.columns, 'No "lemmatized" column found.'
        test_corpus = self.test['lemmatized'].map(lambda x: ' '.join([word for word in x])).tolist()
        train_tfidf = self.tf_fit.transform(self.tf_train_corpus)
        test_tfidf = self.tf_fit.transform(test_corpus)
        return train_tfidf, test_tfidf
    
    def tf_mowe(self):
        """Short summary.
        
        Returns:
        
        """
        assert 'lemmatized' in self.train.columns and 'lemmatized' in self.test.columns, 'No "lemmatized" column found.'
        assert self.word_weights != 'None', 'No word_weights found. Run `tfidf_fit` method.'
        #weights = self.tf_fit.idf_
        idf_word_weights = dict(zip(self.tf_fit.get_feature_names(), weights))
        mowe_idf_train = (np.array([np.mean([self.model[word]*self.word_weights[word] for word in val if word in self.model.wv.vocab] 
                                            or [np.zeros(self.dim)], axis = 0) for idx, val in self.train['lemmatized'].iteritems()]))
        mowe_idf_test = (np.array([np.mean([self.model[word]*self.word_weights[word] for word in val if word in self.model.wv.vocab] 
                                            or [np.zeros(self.dim)], axis = 0) for idx, val in self.test['lemmatized'].iteritems()]))
        return mowe_idf_train, mowe_idf_test

def train_models():
    """short summary.
    
    longer summary.
    
    Returns:
    
    """
    dbow = D2V(X_train, model_type = 'dbow').tag_docs().model_train()
    dbow_fit = GetVectors(X_train, X_test, model = dbow)
    tf_fitted = dbow_fit.tfidf_fit()
    return dbow_fit, tf_fitted

def feature_extraction(dbow, tf):
    """short summary.
    
    longer summary.
    
    Returns:
    
    """
    #PV-DBOW document vectors
    dbow_doc_vecs_train, dbow_doc_vecs_test = dbow.d2v_vecs()
    #MOWE vectors
    dbow_mowe_train, dbow_mowe_test = dbow.w2v_vecs()
    #Tf-idf matrices
    tfidf_train, tfidf_test = tf.tfidf_transform()
    #Tf-MOWE vectors
    dbow_idf_weighted_train, dbow_idf_weighted_test = tf.tf_mowe()
    
    return dbow_doc_vecs_train, dbow_doc_vecs_test, dbow_mowe_train, dbow_mowe_test, tfidf_train, tfidf_test, bow_idf_weighted_train, dbow_idf_weighted_test

def save_features():
    # model_dbow.save('../Dbow Model/reddit_dbow.model')
    # dump(dbow_doc_vecs_train, '../Data/dbow_vecs_train.joblib')
    # dump(dbow_doc_vecs_test, '../Data/dbow_vecs_test.joblib')
    # dump(dbow_mowe_train, '../Data/dbow_mowe_train.joblib')
    # dump(dbow_mowe_test, '../Data/dbow_mowe_test.joblib')
    # dump(tfidf_train, '../Data/tfidf_train.joblib') 
    # dump(tfidf_test, '../Data/tfidf_test.joblib')
    # dump(dbow_idf_weighted_train, '../Data/mowe_idf_train.joblib')
    # dump(dbow_idf_weighted_test, '../Data/mowe_idf_test.joblib')
    return 

if __name__ == '__main__':
    
    data = load_process_text('test_data.pkl')
    
    X_train, X_test, y_train, y_test = train_test_split(data.drop('sub', axis = 1), 
                                                    data['sub'], 
                                                    test_size = .2, 
                                                    random_state = 13, 
                                                    stratify = data['sub'])
    dbow, tf = train_models()
    extract_features = feature_extraction(dbow, tf)
    #save_features()
    print('Done!')
    
#     #Only including PV-DBOW model here
#     dbow = D2V(X_train, model_type = 'dbow').tag_docs().model_train()
#     dbow_fit = GetVectors(X_train, X_test, model = dbow)
#     #PV-DBOW document vectors
#     dbow_doc_vecs_train, dbow_doc_vecs_test = dbow_fit.d2v_vecs()
#     #MOWE vectors
#     dbow_mowe_train, dbow_mowe_test = dbow_fit.w2v_vecs()
#     #Tf-idf
#     tf_fitted = dbow_fit.tfidf_fit()
#     #Tf-idf matrices
#     tfidf_train, tfidf_test = tf_fitted.tfidf_transform()
#     #Tf-MOWE vectors
#     dbow_idf_weighted_train, dbow_idf_weighted_test = tf_fitted.tf_mowe()    