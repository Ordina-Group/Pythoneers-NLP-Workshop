"""Extract the test csv."""
# pylint: disable=C0103

#!/usr/bin/python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def write_series_to_csv(file_name: str, data: Dict[str,pd.Series]) -> None:
    """Write pandas Series to csv file.

    :param file_name: example.csv
    :param data: Dictionary containing columns as keys and Series as values.
    """
    df = pd.DataFrame(data)
    if file_name.endswith(".csv"):
        df.to_csv(file_name, index=False)
    else:
        print("Please use the .csv extension in the filename.")


def main() -> None:
    """Convert data to a csv file."""
    input_data_path = Path.cwd() / "data/sentiment_competition_all.csv"

    # Convert data to dataframe.
    df = pd.read_csv(input_data_path, sep=",", names=["review", "sentiment"])

    # Remove empty rows.
    df.dropna(inplace=True)

    # Convert to array.
    x = df["review"].tolist()

    # Retrieve classification labels.
    y = df["sentiment"]

    # Split into train and test data.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Reformat into dictionary.
    train_data = {"review": x_train, "sentiment": y_train}
    test_data = {"review": x_test, "sentiment": y_test}

    # Convert to pandas dataframe and save into csv file.
    write_series_to_csv(
        file_name="sentiment_competition_train.csv",
        data=train_data,
    )

    write_series_to_csv(file_name="sentiment_competition_test.csv", data=test_data)


if __name__ == "__main__":
    main()
