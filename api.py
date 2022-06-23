import re

from fastapi import FastAPI, UploadFile

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
    sentiment = "positive"  # TODO: dummy
    return {"sentiment": sentiment}


@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> dict:
    _, contents = get_entry_by_id(TABLE_NAME, file_id)
    named_entities = "named_entities"  # TODO: dummy
    return {"named_entities": named_entities}
