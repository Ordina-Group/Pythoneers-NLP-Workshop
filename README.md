# NLP workshop

The main goal of this workshop is to give you a brief introduction of how to 
make an API and how to use it.

##### Table of Contents  
- [Background information](#background-information)
  - [Api](#api)
  - [Machine learning](#machine-learning)
- [Getting started](#getting-started)
  - [Project setup](#project-setup)
- [Exercises](#exercises)
  - [Do a `GET` request](#1-do-a-get-request)
  - [Create a `POST` request](#2-create-a-post-request)
  - [Get all words from a text](#3-get-all-the-words-from-a-text)
  - [Get the sentiment of a text](#4-get-the-sentiment-of-a-text)
  - [Get the named entities of a text](#5-get-the-named-entities-of-a-text)

## Background information
### API
API stands for *Application Programming Interface*. An API is basically a method 
for two applications (or just two pieces of software) to communicate with one
another.

This communication consists of requests. The two types of requests we are going
to use are:

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
(recent) Python version installed on your device. Python 3.7+ is preferred. When 
you have a recent version of Python installed, you have automatically a version 
of the tool `pip` installed

- Check python version:

    `python -V`

#### Project setup
When the right python version is installed, we can open our project and install
all the required packages for this project.
- Create a virtual environment:

    `python -m venv venv`
- Activate the virtual environment for

    - Linux:
    
        `venv/bin/activate` 

    - Windows:
    
        `venv\Scripts\activate`  
- Update `pip` to get its latest version:
    
    `python -m pip install -U pip`

- Install required packages:
    
    `pip install -r requirements.txt` 

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
> future pythoneer" when you visit `http://127.0.0.1:8000` in your browser.

##### Expected response
```
{"message":"Hello Future Pythoneer!"}
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
    
    `python requester.py`
    
Make sure that you have another terminal active, where the API runner is active.

##### Expected response
```
b'{"message":"file successfully uploaded","file_name":"file.txt"}'
200
```

##### Notes
- A part of the code for this request is already given in `api.py`. Just replace
  the `...` with the right code to get the right output.

### 3. Get all the words from a text
#### Part I
> Create a `GET` request  in `api.py` that returns a dict containing the numbers
> of files and a list of the files when you visit `http://127.0.0.1:8000/file`
> in your browser.

##### Function signature
```
@app.get("/file")
def get_all_files() -> dict:
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
def get_file(file_id: int) -> dict:
    ...
```

##### Expected response
```
{"file_name":<file_name>,"contents":"<contents>"}
```

#### Part III
> Create a `GET` request in `api.py` that returns a dict containing all the 
> words from that file when you visit `http://127.0.0.1:8000/file/{id}/words` in 
> your browser.

##### Function signature
```
@app.get("/file/{file_id}/words")
def get_words(file_id: int) -> dict:
    ...
```

##### Expected response
```
{"word_count":<number_of_words>,"unique_words":["<word_1>", "<word_2>", ...]}
```

### 4. Get the sentiment of a text
> Create a `GET` request in `api.py` that returns a dict containing the 
> sentiment of that file when you visit 
> `http://127.0.0.1:8000/file/{id}/sentiment` in your browser.

##### Additional info
- You can use the package `vaderSentiment` to make life a bit easier. This
  package is already installed.
- You need to install NLTK data. In order to do that, you need to open a console
  and type in the following lines:
  
  ```
  >>> import nltk
  >>> nltk.download('punkt')
  ```

##### Function signature
```
@app.get("/file/{file_id}/sentiment")
def get_sentiment(file_id: int) -> dict:
    ...
```

##### Expected response
```
{"sentiment":[{"neg":<score>,"neu":<score>,"pos":<score>,"compound":<score>}]}
```


### 5. Get the named entities of a text
> Create a `GET` request in `api.py` that returns a dict containing all the 
> named entities of the file when you visit 
> `http://127.0.0.1:8000/file/{id}/sentiment` in your browser.

##### Additional info
- You need to make a model that will train the computer. The data to train the
  computer can be found in `data/train.csv`.
- You can write the code for the model in `nlp_model/classifier.py`
- To train the model, just run `python nlp_model/classifier.py` in a terminal.
- Once you have trained the computer, we can connect the model and the API.

##### Function signature
```
@app.get("/file/{file_id}/named_entities")
def get_named_entities(file_id: int) -> dict:
    ...
```

##### Expected response
```
{"named_entities":[["<word_1>","<entity"],["<word_2>","<entity"],...]]}
```

##### Notes
- It is allowed (an highly recommended) to use the internet if you are 
  struggling (for example, check [this link](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)).
- Once the model is linked and an output, try to optimize the model. Add some
  preprocessing steps for example or try a to upload your own text file and see
  what happens.
