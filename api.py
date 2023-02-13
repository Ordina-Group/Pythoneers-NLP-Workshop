"""Module containing the API methods."""
from typing import Any, Dict

from fastapi import FastAPI, UploadFile

from db import (
    add_entry,
    get_all_entries,
    get_entry_by_id,
)

app = FastAPI()


@app.get("/")
def root() -> Dict[str, str]:
    """The root call of the API."""
    return {"message": "Hello Pythoneer!"}


@app.post("/upload_file")
async def upload_file(file: UploadFile) -> Any:
    """Upload a file."""
    message: str = "file successfully uploaded"
    file_name: str = ""

    try:
        contents = await file.read()
        file_name = str(file.filename)
        data = str(contents, "utf-8")
        add_entry([file_name, data])
    except Exception as exc:
        message = str(exc)
    finally:
        await file.close()

    return ...


@app.get("/file")
def get_file() -> Dict[Any, Any]:
    """Get a specific file."""
    ...


@app.get("/file/{file_id}")
def get_all_files(file_id: int) -> Dict[Any, Any]:
    """Get all the files."""
    ...


@app.get("/file/{file_id}/tokens")
def get_tokens(file_id: int) -> Dict[Any, Any]:
    """Get the tokens of a certain file."""
    ...


@app.get("/file/{file_id}/sentiment")
def get_sentiment(file_id: int) -> Dict[Any, Any]:
    """Get the sentiment of a specific file."""
    ...


@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> Dict[Any, Any]:
    """Get the named entities of a specific file."""
    ...
