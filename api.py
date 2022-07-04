import re
import pickle
from pathlib import Path

from nltk import tokenize
from fastapi import FastAPI, UploadFile
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
from db import (
    add_entry,
    get_all_entries,
    get_entry_by_id,
    ENTITIES,
    TABLE_NAME,
)

app = FastAPI()


@app.get("/")
def root() -> dict:
    return {"message": "Hello Future Pythoneer!"}


@app.post("/upload_file")
async def upload_file(file: UploadFile) -> dict:
    message = "file successfully uploaded"
    file_name = ""

    try:
        contents = await file.read()
        file_name = file.filename
        data = str(contents, "utf-8")
        add_entry(TABLE_NAME, entities=ENTITIES, values=[file_name, data])
    except Exception as e:
        message = e
    finally:
        await file.close()

    return {
        "message": message,
        "file_name": file_name,
    }


@app.get("/file")
def get_file() -> dict:
    entries = get_all_entries(TABLE_NAME)

    nr_files = len(entries)
    files = [{"id": rowid, "file_name": file_name} for rowid, file_name, _ in entries]

    return {
        "nr_files": nr_files,
        "files": files,
    }


@app.get("/file/{file_id}")
def get_file(file_id: int) -> dict:
    file_name, contents = get_entry_by_id(TABLE_NAME, file_id)

    return {
        "file_name": file_name,
        "contents": contents,
    }


@app.get("/file/{file_id}/words")
def get_words(file_id: int) -> dict:
    _, contents = get_entry_by_id(TABLE_NAME, file_id)

    contents_without_punctuation = re.sub(r"[^a-zA-Z0-9_ \n]+", "", contents).lower()
    unique_words = sorted(set(contents_without_punctuation.split()))
    word_count = len(unique_words)

    return {
        "word_count": word_count,
        "unique_words": unique_words,
    }


@app.get("/file/{file_id}/letters")
def get_letters(file_id: int) -> dict:
    _, contents = get_entry_by_id(TABLE_NAME, file_id)

    letter_count = sum(c.isalpha() for c in contents)
    
    return {"letter_count": letter_count}


@app.get("/file/{file_id}/sentiment")
def get_sentiment(file_id: int) -> dict:
    _, contents = get_entry_by_id(TABLE_NAME, file_id)

    sentences = tokenize.sent_tokenize(contents)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiments = [sentiment_analyzer.polarity_scores(sentence) for sentence in sentences]

    return {"sentiment": sentiments}


@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> dict:

    # Get text from uploaded file from database.
    _, contents = get_entry_by_id(TABLE_NAME, file_id)

    # Tokenize (split) the sentences in a list of sentences.
    sentences = tokenize.sent_tokenize(contents)

    # Tokenize each sentence into a list of words
    data = []
    for sentence in sentences:
        print(sentence)
        words = tokenize.word_tokenize(sentence)
        data.extend(words)

    # Transform list to dataframe
    df = pd.DataFrame(data, columns=["WORD"])
    
    # Transform dataframe back to scipy object
    input_array = df["WORD"].tolist()

    # Load tfidf vectorizer that was used with the corresponding classifier.
    vec = pickle.load(open("tfidf_vec.pickle", 'rb'))

    # Transform the given input array into tfidf vector.
    # Please note, fit_transform is used to train the vectorizer object.
    # The function transform is used to make it good for classifier input.
    input_vector = vec.transform(input_array)


    # Retrieve model path
    model_path = Path.cwd() / "nlp_model/clf.pickle"

    # Load classifier model
    cls = pickle.load(open(model_path, "rb"))

    # Get prediction based on input of the file.
    ner_tag = cls.predict(input_vector)

    # Combine word and NER tag in list of tuples
    named_entities = [(word, ner_tag) for word, ner_tag in zip(input_array, ner_tag)]


    # Create dictionary for api with the output
    dict = defaultdict(list)
    dict['named_entities'] = named_entities
    return dict
