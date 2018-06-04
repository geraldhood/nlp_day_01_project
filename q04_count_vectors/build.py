
# Default imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from greyatomlib.nlp_day_01_project.q01_load_data.build import q01_load_data
from nltk.tokenize import TreebankWordTokenizer


# Write your solution here


def q04_count_vectors(path, ranges=(1, 2), max_df=0.5, min_df=2):
    data, X_train, X_test, y_train, y_test = q01_load_data(path)
    X_train = pd.Series(X_train)
    vect = CountVectorizer(decode_error='ignore')
    tokenizer = TreebankWordTokenizer()
    vect.set_params(tokenizer=tokenizer.tokenize, stop_words='english', ngram_range=ranges, max_df=max_df,
                    min_df=min_df)
    train_transformed = vect.fit_transform(X_train)
    test_transformed = vect.transform(X_test)
    return train_transformed, test_transformed


