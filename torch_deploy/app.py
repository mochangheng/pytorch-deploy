from typing import Callable, List

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn


app = FastAPI()
model = None
pre = []
post = []

class ModelInput(BaseModel):
    '''Pydantic Model to receive parameters for the /predict endpoint'''
    array: List

def register_model(new_model: nn.Module) -> None:
    '''Set global variable model'''
    global model
    model = new_model

def register_pre(new_pre: List[Callable]) -> None:
    '''Set global variable pre'''
    global pre
    pre = list(new_pre)

def register_post(new_post: List[Callable]) -> None:
    '''Set global variable post'''
    global post
    post = list(new_post)

@app.get("/")
def root():
    # For testing/debugging
    return {"text": "Hello World!"}

@app.post("/predict")
def predict(model_input: ModelInput):
    '''
    View function handling the main /predict endpoint
    '''
    inp = model_input.array
    # Apply all preprocessing functions
    for f in pre:
        inp = f(inp)
    
    # Pass input through model
    tensor = torch.tensor(inp)
    output = model(tensor)

    # Apply all postprocessing functions
    for f in post:
        output = f(output)

    # If torch tensor or numpy array, transform to list so we can pass it back
    if isinstance(output, (np.array, torch.Tensor)):
        output = output.tolist()
    return {"output": output}