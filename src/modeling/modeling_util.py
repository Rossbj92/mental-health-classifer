#Loading/saving data
from joblib import load, dump

#Data manipulation
import numpy as np
import pandas as pd

#Target label encoding
from sklearn.preprocessing import LabelEncoder

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Model evaluation
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_validate, GridSearchCV, KFold, train_test_split

def load_data():
    """Loads train/test files.

    Returns:
        Train/test sets for the Tf-idf, PV-DBOW, MOWE, and IDF-MOWE
        features, as well as the target labels.
    """
    #Tf-idf
    tfidf_train, tfidf_test = (load('../Data/processed/tfidf_train.joblib'),
                               load('../Data/processed/tfidf_test.joblib'))
    #Doc2vec pv-dbow
    dbow_vecs_train, dbow_vecs_test = (load('../Data/processed/dbow_vecs_train.joblib'),
                                   load('../Data/processed/dbow_vecs_test.joblib'))
    #Word2vec
    mowe_train, mowe_test = (load('../Data/processed/dbow_mowe_train.joblib'),
                           load('../Data/processed/dbow_mowe_test.joblib'))
    #Weighted word2vec
    mowe_idf_train, mowe_idf_test = (load('../Data/processed/mowe_idf_train.joblib'),
                                                     load('../Data/processed/mowe_idf_test.joblib'))
    #Labels
    y_train, y_test = (load('../Data/processed/y_train.joblib'),
                       load('../Data/processed/y_test.joblib'))

    return tfidf_train, tfidf_test, dbow_vecs_train, dbow_vecs_test, \
           mowe_train, mowe_test, mowe_idf_train, mowe_idf_test, y_train, y_test

def label_encode(y_train, y_test):
    """Numerically transforms target labels.

    Args:
        y_train, y_test (arr): Arrays containing target labels

    Returns:
        Arrays containing numerically transformed labels.
    """
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test

def train_val_df(features, y_train, models):
    """Performs train/validation split on specified model selection.

    This function uses train/validation splitting to evaluate F1 scores for
    a user-specified set of features and models. Output is formatted into a
    Pandas dataframe.

    Args:
        features (dict): A dictionary with string key labels and scikit-learn-friendly data as values.
        y_train (arr): An array containing numeric target labels
        models (dict): A dictionary with string key labels and scikit-learn model instances.

    Returns:
        A Pandas dataframe with 'Data', 'Model', and 'F1' columns. 'Data' and 'Model'
        values are set from the 'features' and 'y_train' keys, respectively. 'F1' contains scores
        from the train/validation testing.
    """
    master_eval = pd.DataFrame()

    for name, data in features.items():

        X_train_val, X_val , y_train_val, y_val = train_test_split(data, y_train, test_size = .25, random_state = 713)

        for model_name, model in models.items():
            #Training and scoring model
            instance = model
            instance.fit(X_train_val, y_train_val)
            score = f1_score(y_val, instance.predict(X_val), average = 'macro')

            #Updating dataframe
            row = master_eval.shape[0] + 1
            master_eval.loc[row, 'Data'] = name
            master_eval.loc[row, 'Model'] = model_name
            master_eval.loc[row, 'F1'] = np.round(score, 3)
            print('{:.2f}% complete'.format((master_eval.shape[0]/(len(models)*len(features))*100)))

    return master_eval

def kfold_df(features, y_train, models, splits = 5):
    """Performs kfold cross validation on specified model selection.

    This function uses k-fold cross validation to evaluate the F1 scores for
    a user-specified set of features and models. Output is formatted into a
    Pandas dataframe.

    Args:
        features (dict): A dictionary with string key labels and scikit-learn-friendly data as values.
        y_train (arr): An array containing numeric target labels
        models (dict): A dictionary with string key labels and scikit-learn model instances.
        splits (int): Number of cross validation folds. Default = 5.

    Returns:
        A Pandas dataframe with 'Data', 'Model', and 'F1' columns. 'Data' and 'Model'
        values are set from the 'features' and 'y_train' keys, respectively. 'F1' contains
        the average F1 score from all splits.
    """
    master_eval = pd.DataFrame()

    fold_performance_dict = {}

    kf = KFold(n_splits = splits, shuffle = True, random_state = 71)

    for name, feature in features.items():
        for model_name, model in models.items():
            cv = cross_validate(model,
                           feature,
                           y_train,
                           cv = kf,
                           scoring = 'f1_macro')

            #Updating dataframe
            row = master_eval.shape[0] + 1
            master_eval.loc[row, 'Data'] = name
            master_eval.loc[row, 'Model'] = model_name
            master_eval.loc[row, 'F1'] = np.round(np.mean(cv['test_score']), 3)
            print('{:.2f}% complete'.format((master_eval.shape[0]/(len(models)*len(features))*100)))

    return master_eval
