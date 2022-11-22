from typing import Union
from pydantic import BaseModel

from fastapi import FastAPI

from nlp_model import TextClassifier

classifier = TextClassifier()
app = FastAPI(description='NLP classification service')


class Text(BaseModel):
    text: str


@app.get('/ping')
def ping_server():
    return 'Ready!'


@app.post('/classify')
def classify_text(t: Text):
    topic = classifier.predict(t.text)
    return {
        'topic': topic
    }
