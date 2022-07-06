from db import (
    add_entry,
    get_all_entries,
    get_entry_by_id,
    ENTITIES,
    TABLE_NAME,
)
from fastapi import FastAPI, UploadFile
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle


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

    return {"message": message, "file_name": file_name}


@app.get("/file")
def get_file() -> dict:

    entries = get_all_entries('Files')

    files = [{'id': entry[0], 'file_name': entry[1]} for entry in entries]

    # files = []

    # for entry in entries:
    #     files.append({'id': entry[0], 'file_name': entry[1]})

    response = {'nr_files': len(entries), 'files': files}

    return response


@app.get("/file/{file_id}")
def get_file(file_id: int) -> dict:

    entry = get_entry_by_id('Files', file_id)

    return {'file_name': entry[0], 'contents': entry[1]}


@app.get("/file/{file_id}/words")
def get_words(file_id: int) -> dict:

    unique_tokens = set(word_tokenize(get_entry_by_id('Files', file_id)[1]))

    unique_words = [token for token in unique_tokens
                    if token not in ['.,/;:\'"}{[]!@#$%^&*()\\']]

    return {'word_count': len(unique_words),
            'unique_words': unique_words}


@app.get("/file/{file_id}/letters")
def get_letters(file_id: int) -> dict:
    ...


@app.get("/file/{file_id}/sentiment")
def get_sentiment(file_id: int) -> dict:

    sid = SentimentIntensityAnalyzer()

    entry_text = get_entry_by_id('Files', file_id)[1]

    return {'sentiment': [sid.polarity_scores(entry_text)]}


@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> dict:

    with open('./nlp_model/vectorizer.pickle', 'rb') as handler:
        vectorizer = pickle.load(handler)

    with open('./nlp_model/model.pickle', 'rb') as handler:
        model = pickle.load(handler)

    tokens = word_tokenize(get_entry_by_id('Files', file_id)[1])

    data_X = vectorizer.transform(tokens)

    return {'named_entities': [zip(tokens, model.predict(data_X))]}
