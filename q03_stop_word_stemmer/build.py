
from greyatomlib.nlp_day_01_project.q01_load_data.build import q01_load_data
from nltk.corpus import stopwords
import pandas as pd
stop = set(stopwords.words('english'))


from nltk.stem.porter import PorterStemmer


def q03_stop_word_stemmer(path):
    p_stemmer = PorterStemmer()
    data, X_train, X_test, y_train, y_test = q01_load_data(path)
    X_train = pd.Series(X_train).astype(str)
    stop_words = X_train.apply(lambda row: [i for i in row if i not in stop])
    text = []
    for i in range(len(stop_words)):
        tokens = stop_words[i]
        tokens = [p_stemmer.stem(i) for i in tokens]
        text = text + tokens

    return text

