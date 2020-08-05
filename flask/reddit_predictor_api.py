from joblib import load
import spacy
import en_core_web_sm
import numpy as np

logreg = load('./models/final_model.joblib')
tfidf = load('./models/tfidf_model.joblib')
nlp = en_core_web_sm.load()

classes = ['ADHD', 'Anxiety', 'Depression', 'Non-clinical']

def clean_input(text):
    """Preprocesses text for transformation.

    This function formats text to be transformed for tf-idf.
    Text is lower-cased, line break characters are removed,
    and any whitespace is removed. After tokenization, text
    is lemmatized with no stop words or punctuation.

    Args:
        text (str): User entered form text.

    Returns:
        A list of lemmatized words.
    """
    clean_text = (str(text).lower()
                           .replace('\n', '')
                           .strip()
                 )
    tokens = nlp(clean_text)
    lemmas = [tok.lemma_ for tok in tokens if tok.is_punct == False and tok.is_stop == False]
    return lemmas

def transform_input(text):
    """Tf-idf transforms text.

    Args:
        text (str): User-entered form text.

    Returns:
        A sparse tf-idf matrix.
    """
    lemmas = clean_input(text)
    joined = [' '.join(lemmas)]
    return tfidf.transform(joined)

def predict(text):
    """Outputs classes with probabilities.

    Function uses text transformed into a tf-idf sparse matrix
    to predict probabilities of belonging in each class. Classes
    are outputted in descending order of probability.

    Args:
        text (str): User-entered form text.

    Returns:
        A dictionary with class names as keys and probabilities
        as values.
    """

    transformed_text = transform_input(text)
    pred_probs = logreg.predict_proba(transformed_text).flat

    probs = []
    for index in np.argsort(pred_probs)[::-1]:
        prob = {
            'name': classes[index],
            'prob': round(pred_probs[index], 4)
        }
        probs.append(prob)

    return probs
