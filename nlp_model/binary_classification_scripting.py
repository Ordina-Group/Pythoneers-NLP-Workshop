#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


def main():
    # Retrieve data
    input_data_path = Path.cwd() / "data/sentiment_competition_train.csv"
    output_model_path = Path.cwd() / "binary_clf.pickle"
    output_vectorizer_path = Path.cwd() / "binary_tfidf_vec.pickle"

    # Convert data to dataframe
    df = pd.read_csv(input_data_path, sep=",", names=["remove", "review", "sentiment"]
    df = df.drop(columns="remove")

    # Adjust to allow more data to be trained
    # df = df[:180000]

    # Remove empty rows
    df.dropna(inplace=True)

    # Retrieve classification features
    x = df["review"]

    # Retrieve classification labels
    y = df["sentiment"]

    # Option 1
    # Define classifier and vectorizer
    cls = LogisticRegression()
    vec = TfidfVectorizer()

    # Convert to array
    x = df["review"].tolist()

    # Convert to vector
    x = vec.fit_transform(x)

    # Retrieve classification labels
    y = df["sentiment"]

    # Split into train and test data. Please note, validation data is not used. It splits on the training data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(type(x_test))

    # Train classifier based on training data with corresponding classification labels.
    cls.fit(x_train, y_train)

    # Get classification label predictions.
    y_pred = cls.predict(x_test)

    # Print statistics
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    # Save classifier and corresponding vectorizer for api usage.
    pickle.dump(cls, open(output_model_path, "wb"))
    pickle.dump(vec, open(output_vectorizer_path, "wb"))


if __name__ == "__main__":
    main()
