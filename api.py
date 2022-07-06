from fastapi import FastAPI, UploadFile
# import nltk
# nltk.download('punkt')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


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

    return {'message':message,'file_name':file_name}


@app.get("/file")
def get_file() -> dict:
    files = get_all_entries(TABLE_NAME)
    nr_files = len(files)
    dictlist = []
    for file in files:
        filedict = {}
        filedict['id'] = file[0]
        filedict['filename'] = file[1]
        dictlist.append(filedict)

    return {'nr_files': nr_files, 'files': dictlist}


@app.get("/file/{file_id}")
def get_file(file_id: int) -> dict:
    file = get_entry_by_id(TABLE_NAME, file_id)


    return {'file_name': file[0], 'contents':file[1]}

@app.get("/file/{file_id}/words")
def get_words(file_id: int) -> dict:
    file = get_entry_by_id(TABLE_NAME, file_id)
    contents = file[1]
    word_list = contents.split()
    word_count = len(word_list)

    return {'word_count':word_count,'word_list':set(word_list)}


@app.get("/file/{file_id}/letters")
def get_letters(file_id: int) -> dict:
    ...


@app.get("/file/{file_id}/sentiment")

def get_sentiment(file_id: int) -> dict:
    file = get_entry_by_id(TABLE_NAME, file_id)
    contents = file[1]
    sent_ana = SentimentIntensityAnalyzer()
    score = sent_ana.polarity_scores(contents)
    return {'sentiment':[score]}


@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> dict:
    file = get_entry_by_id(TABLE_NAME, file_id)
    text = file[1]
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForTokenClassification.from_pretrained("malduwais/distilbert-base-uncased-finetuned-ner")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(text)
    print({'ner_results': ner_results})
    #return({'named_entities':})
