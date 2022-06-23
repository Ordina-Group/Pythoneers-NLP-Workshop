# NLP-Workshop

## Setup
- Create virtual environment:

    `python -m venv venv`
- Activate virtual environment:

    `source venv/bin/activate` (Linux)

    `source venv/Scripts/activate` (Windows) 
- Update pip:
    
    `python -m pip install -U pip`
- Install required packages:
    
    `python -m install -r requirements.txt`

## Usage
- Setup the database:

    `python db.py`

- Run server:
    
    `uvicorn api:app --reload`

-  Perform POST request:

    `python requester.py`
