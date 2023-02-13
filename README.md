# NLP workshop

The main goal of this workshop is to give an introduction into NLP. Hopefully,
this will both be interesting and spark ideas for many professional and/or
personal projects!

##### Table of Contents  
- [Background information](#background-information)
  - [API](#api)
  - [Machine learning](#machine-learning)
- [Getting started](#getting-started)
  - [Project setup](#project-setup)
  - [Check Python version](#check-python-version)
- [Exercises](#exercises)
  1. [Do a `GET` request](#1-do-a-get-request)
  2. [Create a `POST` request](#2-create-a-post-request)
  3. [Get all tokens from a text](#3-get-all-the-tokens-from-a-text)
  4. [Get the sentiment of a text](#4-get-the-sentiment-of-sentence-tokens-of-all-the-lines-of-a-text-file)
  5. [Get the named entities of a text](#5-get-the-named-entities-of-a-text)
- [Competition](#competition)

## Background information
### API
API stands for *Application Programming Interface*. An API is basically a method 
for two applications (or just two pieces of software) to communicate with one
another.

This communication can take place in many different forms, but often
when talking about APIs we mean web APIs. This communication consists of
requests. The two types of requests we are going to use are:

- `GET` requests: a request to get some data from the resource
- `POST` request: a request to send some data to the resource

We are gonna use `FastAPI`, which is a framework for building such API's. See
[this link](https://fastapi.tiangolo.com/tutorial/) for some more info.

### Machine learning

Machine learning is a way to make the computer learn something. The computer can
learn from studying sorts of data and statistics. It is a program that can
predict an outcome after it learned and analysed the data.

For NLP specifically, machine learning ensures that the computer can identify
certain aspects of text, such as speech or entities. This technique uses a model 
that is then applied to a certain text, which will train the computer.

For more information, you can use [this link](https://www.analyticsvidhya.com/blog/2021/07/nltk-a-beginners-hands-on-guide-to-natural-language-processing/).
Oh! Don't forget that Google is your friend!

## Getting started
Before we can make a new Python project, we need to make sure that you have a 
(recent) Python version installed on your device. Python 3.8+ is required. If 
you have a recent version of Python installed, a version of the tool `pip` is
automatically installed.

#### Check Python version
`python --version`

**Note**: When the command is not found or the version is 2.x, try `python3 --version` and use `python3` instead of `python` in the commands below.

#### Project setup
When the right python version is installed, we can open our project and install
all the required packages for this project.
- Create a virtual environment:

    `python -m venv venv`
- Activate the virtual environment:

    - See [this link](https://docs.python.org/3/library/venv.html) how to activate the `venv` for your operating system.
    
- Update `pip` to get its latest version:
    
    `python -m pip install -U pip`

- Install wheel:
    
    `python -m pip install wheel` 


- Install required packages:
    
    `python -m pip install -r requirements.txt` 

- Set up the database:

    `python db.py`
    
- Run the server:

    `uvicorn api:app --reload`
    
    **Note**: This "runner" will use a terminal instance while active. If you 
    want to run an other command, just open a new terminal instance (and don't 
    kill this one).

## Exercises
### 1. Do a `GET` request
> Do a `GET` request that returns a dictionary containing the message "Hello 
> Future Pythoneer" when you visit `http://127.0.0.1:8000` in your browser.

##### Expected response
```
{"message":"Hello Pythoneer!"}
```

##### Notes
- The code for this request is already given in `api.py`.

### 2. Create a `POST` request
> Create a `POST` request that uploads a file when you execute 
> `python requester.py` in your terminal.

##### Additional info
The post request uses the module `requester.py` (just take a look and see what
happens there). To do the actual upload of files to the database, run the 
following command:
    
    python requester.py
    
Make sure that you have another terminal active, where the API runner is active.

##### Expected response
```
b'{"message":"file successfully uploaded","file_name":"file.txt"}'
200
```

##### Notes
- A part of the code for this request is already given in `api.py`. Just replace
  the `...` with the right code to get the right output.

### 3. Get all the tokens from a text
#### Part I
> Create a `GET` request  in `api.py` that returns a dict containing the numbers
> of files and a list of the files when you visit `http://127.0.0.1:8000/file`
> in your browser.

##### Function signature
```
@app.get("/file")
def get_all_files() -> Dict[str, Any]:
    ...
```

##### Expected response
```
{"nr_files":<number_of_files>,"files":[{"id":<id>,"file_name":"<file_name>"},...}]}
```

#### Part II
> Create a `GET` request in `api.py` that returns a dict containing the file
> name and its content when you visit `http://127.0.0.1:8000/file/{id}` in your
> browser.

##### Function signature
```
@app.get("/file/{file_id}")
def get_file(file_id: int) -> Dict[str, str]:
    ...
```

##### Expected response
```
{"file_name":<file_name>,"contents":"<contents>"}
```

#### Part III
> Create a `GET` request in `api.py` that returns a dict containing all the 
> tokens from that file when you visit `http://127.0.0.1:8000/file/{id}/words` in 
> your browser. A token can be a set of multiple words that belong together like 'New York' or 'Harry Potter'.

##### Function signature
```
@app.get("/file/{file_id}/tokens")
def get_tokens(file_id: int) -> Dict[str, Any]:
    ...
```

##### Expected response
```
{"token_count":<number_of_tokens>,"unique_tokens":["<token_1>", "<token_2>", ...]}
```

### 4. Get the sentiment of sentence tokens of all the lines of a text file. 
> Create a `GET` request in `api.py` that returns a dict of sentence tokens containing the 
> sentiment of that file when you visit 
> `http://127.0.0.1:8000/file/{id}/sentiment` in your browser. A line can hold multiple sentence tokens.
> For example, the line below holds three sentence tokens :
> 
>  "This is the first sentence. This is the second sentence. This is the third sentence."

##### Function signature
```
@app.get("/file/{file_id}/sentiment")
def get_sentiment(file_id: int) -> Dict[str, Any]:
    ...
```

##### Expected response
```
{"sentiment":{<sentence_text>:{"neg":<score>,"neu":<score>,"pos":<score>,"compound":<score>},...}}
```

### 5. Get the named entities of a text
> Create a `GET` request in `api.py` that returns a dict containing all the 
> named entities of the file when you visit 
> `http://127.0.0.1:8000/file/{id}/sentiment` in your browser.

##### Function signature
```
@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> Dict[str, Any]:
    ...
```

##### Expected response
```
{"named_entities":[["<token_1>","<entity_tag>"],["<token_2>","<entity_tag>"],...]]}
```

## Competition
Now it is time to train your own NLP model! 
You are going to perform binary sentiment classification which means classifying the sentiment of a review to either positive or negative (0 or 1).

These steps can be followed as a reference:
- The data to train your model can be found in `data\sentiment_competition_train.csv`
- Pre-process dataset
- Split dataset into a training and validation set
- Vectorize data
- Train model using classification algorithm
- Validate trained model using validation dataset
- Improve model/pre-processing/vectorizer etc.
- Evaluate with the test set and hope for the best!
