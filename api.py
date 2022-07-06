import nltk
from fastapi import FastAPI, UploadFile, HTTPException
from nltk import RegexpTokenizer, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

from db import (
    add_entry,
    get_all_entries,
    get_entry_by_id,
    ENTITIES,
    TABLE_NAME,
)
from models.message import Message

app = FastAPI(
    title="Ordina-hashpipe text analysis API",
    description="This api offers advanced text analysis functions",
    version="1.0.0",
    terms_of_service="https://www.ordina.nl/disclaimer/",
    contact={
        "name": "Hashpipe",
        "url": "https://www.ordina.nl/contact/",
        "email": "info@ordina.nl",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.on_event("startup")
async def startup_event():
    nltk.download('punkt')
    nltk.download('vader_lexicon')


@app.get("/", name="Entry point for the application", description="This is the entry point for the application")
def root() -> dict:
    return {"message": "Hello Future Pythoneer!"}


@app.post("/upload_file",
          name="Upload a file",
          description="With this endpoint you can upload a file to the server so you can analyse it.",
          responses={404: {"model": Message}})
async def upload_file(file: UploadFile) -> dict:
    message: str = "file successfully uploaded"
    file_name: str = ""

    try:
        contents: bytes = await file.read()
        file_name = file.filename
        data: str = str(contents, "utf-8")
        add_entry(TABLE_NAME, entities=ENTITIES, values=[file_name, data])
    except Exception as e:
        message = str(e)
    finally:
        await file.close()

    return {"message": message, "file_name": file_name}


@app.get("/file",
         name="Get all files",
         description="With this endpoint you can get all the files currently in the database.",
         responses={404: {"model": Message}})
def get_file() -> dict:
    entries = get_all_entries("FILES")
    result: list = []

    for entry in entries:
        new_entry: dict = {
            "id": entry[0],
            "file_name": entry[1]
        }

        result.append(new_entry)

    return {"nr_files": len(entries), "files": result}


@app.get("/file/{file_id}",
         name="Get file",
         description="With this endpoint you can get an specific file with its contents.",
         responses={404: {"model": Message}})
def get_file(file_id: int) -> dict:
    result: tuple = get_entry_by_id("FILES", file_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Item not found")

    return {"file_name": result[0], "contents": result[1]}


@app.get("/file/{file_id}/words",
         name="Get words in file",
         description="With this endpoint you can get the word count, the unique word count and a list of unique words "
                     "for a file",
         responses={404: {"model": Message}})
def get_words(file_id: int) -> dict:
    result: tuple = get_entry_by_id("FILES", file_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Item not found")

    text: str = result[1]
    tokenizer: RegexpTokenizer = RegexpTokenizer(r'\w+')
    results: list[str] = tokenizer.tokenize(text.lower())

    return {"word_count": len(results), "unique_word_count": len(set(results)), "unique_words": results}


@app.get("/file/{file_id}/letters",
         name="Get letter count",
         description="With this endpoint you can the amount of unique and non-unique letters of a file.",
         responses={404: {"model": Message}})
def get_letters(file_id: int) -> dict:
    result: tuple = get_entry_by_id("FILES", file_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Item not found")

    text: str = result[1]
    tokenizer: RegexpTokenizer = RegexpTokenizer(r'\w+')
    results = tokenizer.tokenize(text.lower())

    all_letters: list[chr] = []
    total_length: int = 0

    for word in results:
        for letter in word:
            all_letters += letter
        total_length += len(word)

    return {"amount_of_letters": total_length, "unique_letters": len(set(all_letters))}


@app.get("/file/{file_id}/sentiment",
         name="Get sentiment",
         description="With this endpoint you can get the sentiment of all the sentences in a file",
         responses={404: {"model": Message}})
def get_sentiment(file_id: int) -> dict:
    result: tuple = get_entry_by_id("FILES", file_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Item not found")

    text: str = result[1]
    text: str = text.replace("\r\n", " ")
    tokenized_text: list[str] = sent_tokenize(text)
    sentiment_results: list = []

    sid: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()

    for sentence in tokenized_text:
        sentiment_score: dict[str, float] = sid.polarity_scores(sentence)
        sentiment: dict = {"neg": sentiment_score["neg"], "neu": sentiment_score["neu"], "pos": sentiment_score["pos"],
                           "compound": sentiment_score["compound"], "sentence": sentence}

        sentiment_results.append(sentiment)

    return {"sentiment": sentiment_results}


@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> dict:
    result: tuple = get_entry_by_id("FILES", file_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Item not found")

    text: str = result[1]
    tokenizer: RegexpTokenizer = RegexpTokenizer(r'\w+')
    results = tokenizer.tokenize(text.lower())

    for word in results:
        print(word)


    return {"named_entities": [["<word_1>", "<entity"]]}
