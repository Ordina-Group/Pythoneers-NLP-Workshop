import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


def main():
    df: DataFrame = pd.read_csv('../data/train.csv', sep="\t")
    df_dev: DataFrame = pd.read_csv('../data/dev.csv', sep="\t")

    df = df[:180000]

    df.rename(columns={'0': 'Sentence number', '-DOCSTART-': 'woord', 'pos': 'POS', 'pos2': 'POS2', 'label': 'label'},
              inplace=True)

    df_dev.rename(
        columns={'0': 'Sentence number', '-DOCSTART-': 'woord', 'pos': 'POS', 'pos2': 'POS2', 'label': 'label'},
        inplace=True)

    df = df.replace(to_replace='None', value=np.nan).dropna()

    cls = LogisticRegression()
    vec = TfidfVectorizer()

    x = df["woord"].tolist()

    x = vec.fit_transform(x)

    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    cls.fit(x_train, y_train)

    y_pred = cls.predict(x_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, cls.predict(x_test)))

    pickle.dump(cls, open("clf.pickle", "wb"))
    pickle.dump(vec, open("vec.pickle", "wb"))


if __name__ == '__main__':
    main()
