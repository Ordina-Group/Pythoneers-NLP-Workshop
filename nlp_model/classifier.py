import pickle
from pathlib import Path

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

def main():
    input_data_path = Path.cwd() / "data/train.csv"
    output_model_path = Path.cwd() / "nlp_model/clf.pickle"
    output_vectorizer_path = Path.cwd() / "nlp_model/tfidf_vec.pickle"

    train = pd.read_csv(input_data_path, sep="\t", names=["SENTENCE_NR", "WORD", "POS", "POS_TAG", "NER_TAG"])
    
    train = train[:180000]
    train.dropna(inplace=True)
    
    X = train["WORD"].tolist()
    Y = train["NER_TAG"].tolist()

    model = LinearSVC()
    vec = TfidfVectorizer()

    X = vec.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, model.predict(x_test)))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    pickle.dump(model, open(output_model_path, "wb"))
    pickle.dump(vec, open(output_vectorizer_path, "wb"))

if __name__ == '__main__':
    main()
