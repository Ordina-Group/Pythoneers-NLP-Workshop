from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

"""Run this file once to create model and save it to pickle"""


def split_labels(df):
    data_X = df.token.tolist()
    data_y = df.ner.tolist()
    return data_X, data_y


def lr_model(train_X, train_y):
    model = LogisticRegression(random_state=0, max_iter=10000)
    model.fit(train_X, train_y)
    return model


def save_to_pickle(model, vectorizer):

    with open('./model.pickle', 'wb') as handler:
        pickle.dump(model, handler, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./vectorizer.pickle', 'wb') as handler:
        pickle.dump(vectorizer, handler, protocol=pickle.HIGHEST_PROTOCOL)


def main():

    df = pd.read_csv('./../data/train.csv', sep='\t', header=0, na_values='O',
                     names=['sent_no', 'token', 'pos', 'pos_tag', 'ner'])

    df.dropna(inplace=True)

    data_X, data_y = split_labels(df)

    vectorizer = TfidfVectorizer()

    train_vectors = vectorizer.fit_transform(data_X)

    model = lr_model(train_vectors, data_y)

    print(model.predict(vectorizer.transform(
        ['This', 'is', 'a', 'list', 'EU'])))

    #save_to_pickle(model, vectorizer)


if __name__ == '__main__':
    main()
