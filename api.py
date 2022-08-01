from fastapi import FastAPI, UploadFile
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
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

    return message


@app.get("/file")
def get_file() -> dict:
    # {"nr_files":<number_of_files>,"files":[{"id":<id>,"file_name":"<file_name>"},...}]}
    entries = get_all_entries(TABLE_NAME)
    nr_files = len(entries)
    files = []
    for item in entries:
        files.append({item[0]: item[1]})
    file_dict = {'nr_files': nr_files, 'files':files}
    return file_dict



@app.get("/file/{file_id}")
def get_file(file_id: int) -> dict:
    entry = get_entry_by_id(TABLE_NAME, file_id)
    return {'file_name': entry[0], 'contents': entry[1]}


@app.get("/file/{file_id}/words")
def get_words(file_id: int) -> dict:
    entry = get_entry_by_id(TABLE_NAME, file_id)
    text = entry[1]

    tokens = nltk.tokenize.word_tokenize(text)
    return {"word_count": len(tokens), "unique_words": set(tokens)}


@app.get("/file/{file_id}/letters")
def get_letters(file_id: int) -> dict:
    ...


@app.get("/file/{file_id}/sentiment")
def get_sentiment(file_id: int) -> dict:
    entry = get_entry_by_id(TABLE_NAME, file_id)
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = analyzer.polarity_scores(entry[1])
    return {'sentiment': sentiment_dict}


@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> dict:
    entry = get_entry_by_id(TABLE_NAME, file_id)
    text = entry[1]
    model = spacy.load("en_core_web_sm")
    doc = model(text)
    pos_list = []
    for word in doc.ents:
        pos_list.append([word.text, word.label_])
    return {'named_entities': pos_list}



