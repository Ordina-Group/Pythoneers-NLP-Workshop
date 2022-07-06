import re
import pickle
from pathlib import Path
from collections import defaultdict
import pandas as pd

from fastapi import FastAPI, UploadFile
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string

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

    return {"message": message, "file_name":file_name}


@app.get("/file")
def get_file() -> dict:
    files = get_all_entries(TABLE_NAME)
    files = [{'id': f[0], 'file_name': f[1]} for f in files]
    return {"nr_files":len(files), "files": files}


@app.get("/file/{file_id}")
def get_file(file_id: int) -> dict:
    file = get_entry_by_id(TABLE_NAME, file_id)
    return {"file_name": file[0], "contents": file[1]}


@app.get("/file/{file_id}/words")
def get_words(file_id: int) -> dict:
    file = get_entry_by_id(TABLE_NAME, file_id)
    translator = str.maketrans('', '', string.punctuation)
    words = file[1].translate(translator)
    words = word_tokenize(words)
    return {"word_count": len(words), "unique_words": set(words)}


@app.get("/file/{file_id}/letters")
def get_letters(file_id: int) -> dict:
    ...


@app.get("/file/{file_id}/sentiment")
def get_sentiment(file_id: int) -> dict:
    analyzer = SentimentIntensityAnalyzer()

    file = get_entry_by_id(TABLE_NAME, file_id)
    sentences = sent_tokenize(file[1])
    sentiments = []
    for sentence in sentences:
        polarity =  analyzer.polarity_scores(sentence)
        sentiments.append(polarity)
    return {"sentiment": sentiments}



@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> dict:
    _, contents = get_entry_by_id(TABLE_NAME, file_id)

    sentences = sent_tokenize(contents)

    data = []
    for sentence in sentences:
        print(sentence)
        words = word_tokenize(sentence)
        data.extend(words)

    df = pd.DataFrame(data, columns=["WORD"])

    input_array = df["WORD"].tolist()

    vec = pickle.load(open("nlp_model/tfidf_vec.pickle", 'rb'))

    input_vector = vec.transform(input_array)


    model_path = Path.cwd() / "nlp_model/clf.pickle"

    cls = pickle.load(open(model_path, "rb"))

    ner_tag = cls.predict(input_vector)

    named_entities = [(word, ner_tag) for word, ner_tag in zip(input_array, ner_tag)]

    dict = defaultdict(list)
    dict['named_entities'] = named_entities
    return dict
