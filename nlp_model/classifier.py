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
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def main():
    input_data_path = Path.cwd() / "data/train.csv"
    output_model_path = Path.cwd() / "nlp_model/clf.pickle"

    df = pd.read_csv(input_data_path, sep='\t', names=['SENTENCE_NR', 'WORD', 'POS', 'POS_TAG', 'NER_TAG'])

    # Adjust to allow more data to be trained
    df = df[:5000]

    print(df.head())

    df.dropna(inplace=True)

    cls = LogisticRegression()
    vec = TfidfVectorizer()

    classifier = Pipeline([("vec", vec), ("cls", cls)])


    # x = df[["SENTENCE_NR", "WORD", "POS", "POS_TAG"]].tolist()
    x = df["WORD"].tolist()
    x = vec.fit_transform(x).toarray()

    y = df["NER_TAG"]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


    print("shape")
    print(x.shape)
    print(y.shape)

    print(x_train.shape)
    print(y_train.shape)



    print("going to fit ")
    print(y_train)

    cls.fit(x_train, y_train)
    y_pred = cls.predict(x_test)
    print(y_pred)
    print(classification_report(y_test, cls.predict(x_test)))

    pickle.dump(cls, open(output_model_path, "wb"))


if __name__ == '__main__':
    main()
