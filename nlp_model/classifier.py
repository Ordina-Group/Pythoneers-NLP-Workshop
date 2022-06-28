#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main():
    df = pd.read_csv("../data/train.csv", sep='\t', names=['SENTENCE_NR', 'WORD', 'POS', 'POS_TAG', 'NER_TAG'])
    
    # Adjust to allow more data to be trained
    df = df[:5000]

    print(df.head())

    df.dropna(inplace=True)

    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform(df.to_dict("records"))

    y = df.NER_TAG.values
    all_classes = np.unique(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    clf = Perceptron(verbose=10, n_jobs=-1)
    clf.partial_fit(x_train, y_train, all_classes)

    print(classification_report(y_test, clf.predict(x_test)))
    print(f1_score(clf.predict(x_test), y_test, average="micro"))


if __name__ == '__main__':
    main()
