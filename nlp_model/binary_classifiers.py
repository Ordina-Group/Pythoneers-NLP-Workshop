import logging
from pathlib import Path

import pandas as pd
import sklearn
from sklearn import linear_model, naive_bayes
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class BinaryDataProcessor:
    processed = False
    x_train = None
    x_validation = None
    y_train = None
    y_validation = None

    def __init__(self, dataframe: pd.DataFrame, vectorizer: text) -> None:
        """Set initial input and converter for processing."""
        self.df = dataframe
        self.vec = vectorizer()

    def __repr__(self) -> str:
        return f"BinaryDataProcessor: Vectorizer={self.vec}"

    def pre_process(self, x_column: str, y_column: str, ) -> None:
        """Pre-process dataframe.

        1. Remove empty rows.
        2. Get features using the x column of the dataframe (array).
        3. Get labels using the y column of the dataframe (array).
        4. Split into train and test data.
        """
        self.df.dropna(inplace=True)
        x = self.df[x_column].tolist()
        y = self.df[y_column]
        x = self.vec.fit_transform(x)
        self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(x,
                                                                                            y,
                                                                                            test_size=0.2,
                                                                                            random_state=0)
        self.processed = True
        logger.info("Processed dataframe with: x_column=%s and y_column=%s" % (x_column, y_column))


class BinaryModelProcessor:
    fitted = False

    def __init__(self, data_processor: BinaryDataProcessor, model: sklearn) -> None:
        self.data_processor = data_processor
        self.model = model()

    def __repr__(self) -> str:
        return f"BinaryModelProcessor: Model={self.model}"

    def fit_model(self) -> None:
        """Train classifier

        The model is fitted based on training data with corresponding
        classification labels.
        """
        if self.data_processor.processed:
            self.model.fit(self.data_processor.x_train, self.data_processor.y_train)
            logger.info("%s processed using data processor: %s" % (self, self.data_processor))
        else:
            logger.warning("Data processor still needs processing: %s" % self.data_processor)

    def validate_model(self) -> None:
        """Validate model based on predicted labels versus validated labels."""
        y_predictions = self.model.predict(self.data_processor.x_validation)
        logger.info(
            "Confusion matrix:\n\n %s" % metrics.confusion_matrix(self.data_processor.y_validation,
                                                                  y_predictions))
        logger.info(metrics.classification_report(self.data_processor.y_validation, y_predictions))
        logger.info("Accuracy: %s" % metrics.accuracy_score(self.data_processor.y_validation, y_predictions))


def get_binary_data_processing_strategies(dataframe: pd.DataFrame) -> dict:
    """Returns a dictionary with multiple binary data processing strategies."""
    strategies = {"tf-idf": BinaryDataProcessor(dataframe=dataframe,
                                                vectorizer=text.TfidfVectorizer),
                  "count": BinaryDataProcessor(dataframe=dataframe,
                                               vectorizer=text.CountVectorizer),
                  "hashing": BinaryDataProcessor(dataframe=dataframe,
                                                 vectorizer=text.HashingVectorizer),
                  }
    return strategies


def get_binary_model_classification_strategies(data_processor: BinaryDataProcessor) -> dict:
    """Returns a dictionary with multiple binary model classification strategies."""
    strategies = {"logistic_regression": BinaryModelProcessor(data_processor=data_processor,
                                                              model=linear_model.LogisticRegression),
                  "support_vector_machine": BinaryModelProcessor(data_processor=data_processor,
                                                                 model=svm.SVC),
                  "naive_bayes": BinaryModelProcessor(data_processor=data_processor,
                                                      model=naive_bayes.GaussianNB),
                  }
    return strategies


def main():
    # Retrieve data
    input_data_path = Path.cwd() / "../data/sentiment_competition_dataset.csv"

    # Convert data to dataframe
    df = pd.read_csv(input_data_path, sep=",", names=["review", "sentiment"])

    binary_data_strategies = get_binary_data_processing_strategies(dataframe=df)
    binary_model_strategies = []

    features = "review"
    label = "sentiment"

    for data_processor in binary_data_strategies.values():
        data_processor.pre_process(x_column=features, y_column=label)
        binary_model_strategies.append(get_binary_model_classification_strategies(data_processor))

    for strategy in binary_model_strategies:
        for model_processor in strategy.values():
            print(model_processor)
            model_processor.fit_model()
            model_processor.validate_model()


if __name__ == "__main__":
    main()