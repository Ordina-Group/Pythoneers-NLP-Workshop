"""Module containing the API methods."""
from typing import Dict, Any, List, Union

from fastapi import FastAPI, UploadFile
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from db import (
    add_entry,
    get_all_entries,
    get_entry_by_id,
)

nltk.download("words")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")

app = FastAPI()


@app.get("/")
def root() -> Dict[str, str]:
    """The root call of the API."""
    return {"message": "Hello Pythoneer!"}


@app.post("/upload_file")
async def upload_file(file: UploadFile) -> Dict[str, str]:
    """Upload a file."""
    message = "file successfully uploaded"
    file_name = ""

    try:
        contents = await file.read()
        file_name = file.filename
        data = str(contents, "utf-8")
        add_entry([file_name, data])
    except Exception as exc:
        message = str(exc)
    finally:
        await file.close()

    return {
        "message": message,
        "file_name": file_name,
    }


@app.get("/file")
def get_all_files() -> Dict[str, Any]:
    """Get all the files."""
    entries = get_all_entries()
    nr_files = len(entries)
    files = [{"id": rowid, "file_name": file_name} for rowid, file_name in entries]

    return {
        "nr_files": nr_files,
        "files": files,
    }


@app.get("/file/{file_id}")
def get_file(file_id: int) -> Dict[str, str]:
    """Get a specific file."""
    file_name, contents = get_entry_by_id(file_id)

    return {
        "file_name": file_name,
        "contents": contents,
    }


@app.get("/file/{file_id}/tokens")
def get_tokens(file_id: int) -> Dict[str, Any]:
    """Get the tokens of a certain file."""
    _, contents = get_entry_by_id(file_id)

    tokens = nltk.tokenize.word_tokenize(contents)
    unique_tokens = sorted(set(token.lower() for token in tokens if token.isalpha()))
    token_count = len(unique_tokens)

    return {
        "token_count": token_count,
        "unique_tokens": unique_tokens,
    }


@app.get("/file/{file_id}/sentiment")
def get_sentiment(file_id: int) -> Dict[str, Dict[str, Any]]:
    """Get the sentiment of a specific file."""
    _, contents = get_entry_by_id(file_id)

    sentences = nltk.tokenize.sent_tokenize(contents)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiments = {s: sentiment_analyzer.polarity_scores(s) for s in sentences}

    return {"sentiment": sentiments}


@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> Dict[str, List[List[Any]]]:
    """Get the named entities of a specific file."""
    _, contents = get_entry_by_id(file_id)

    tokens = nltk.tokenize.word_tokenize(contents)
    tagged_tokens = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(tagged_tokens)
    named_entities = [
        [x.leaves()[0][0], x.label()] for x in chunks if hasattr(x, "label")
    ]

    return {"named_entities": named_entities}
