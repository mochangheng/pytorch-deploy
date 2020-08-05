from typing import Callable, List

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn


app = FastAPI()
model = None
pre = []
post = []

class ModelInput(BaseModel):
    array: List

def register_model(new_model: nn.Module) -> None:
    global model
    model = new_model

def register_pre(new_pre: List[Callable]) -> None:
    global pre
    pre = list(new_pre)

def register_post(new_post: List[Callable]) -> None:
    global post
    post = list(new_post)

@app.get("/")
def root():
    return {"text": "Hello World!"}

@app.post("/predict")
def predict(model_input: ModelInput):
    tensor = torch.tensor(model_input.array)
    output = model(tensor)
    return {"output": output.tolist()}