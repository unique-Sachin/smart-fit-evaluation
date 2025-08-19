from rag import query_about_nutrition
from typing import Union
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.get("/query")
def query_rag(query:str):
    res = query_about_nutrition(query=query)
    return res

