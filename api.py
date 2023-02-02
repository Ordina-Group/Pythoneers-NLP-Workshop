from fastapi import FastAPI, UploadFile

from db import (
    add_entry,
    get_all_entries,
    get_entry_by_id,
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
        add_entry([file_name, data])
    except Exception as e:
        message = str(e)
    finally:
        await file.close()

    return ...


@app.get("/file")
def get_file() -> dict:
    ...


@app.get("/file/{file_id}")
def get_all_files(file_id: int) -> dict:
    ...


@app.get("/file/{file_id}/tokens")
def get_tokens(file_id: int) -> dict:
    ...


@app.get("/file/{file_id}/sentiment")
def get_sentiment(file_id: int) -> dict:
    ...


@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> dict:
    ...
