#!/usr/bin/python
# -*- coding: utf-8 -*-

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def main():
    input_data_path = Path.cwd() / "data/train.csv"
    output_model_path = Path.cwd() / "nlp_model/clf.pickle"

    df = pd.read_csv(input_data_path, sep='\t', names=['SENTENCE_NR', 'WORD', 'POS', 'POS_TAG', 'NER_TAG'])
    
    # df = df[["SENTENCE_NR", "WORD"]]

    # Adjust to allow more data to be trained
    df = df[:5000]

    print(df.head())

    df.dropna(inplace=True)

    # vectorizer = DictVectorizer(sparse=False)
    # x = vectorizer.fit_transform(df.to_dict("records"))
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=0, stop_words="english")
    x = vectorizer.fit_transform(df["WORD"])

    vectorizer = CountVectorizer(stop_words="english")
    x_new = vectorizer.fit(df["WORD"])

    y = df.NER_TAG.values
    all_classes = np.unique(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # clf = Perceptron(verbose=10, n_jobs=-1)
    # clf.partial_fit(x_train, y_train, all_classes)

    # print(classification_report(y_test, clf.predict(x_test)))
    # print(f1_score(clf.predict(x_test), y_test, average="micro"))

    # pickle.dump(clf, open(output_model_path, "wb"))

    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    print(classification_report(y_test, log_reg.predict(x_test)))

    pickle.dump(log_reg, open(output_model_path, "wb"))


if __name__ == '__main__':
    main()
