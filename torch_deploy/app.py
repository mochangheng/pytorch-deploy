from typing import Callable, List, Dict, Union

from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import sys

from .logger import Logger

app = FastAPI()
model = None
pre = []
post = []
logger = None


class ModelInput(BaseModel):
    '''Pydantic Model to receive parameters for the /predict endpoint'''
    inputs: Union[List, Dict]

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

def create_logger(log_file: str) -> None:
    global logger
    logger = Logger(log_file)

@app.get("/")
def root():
    # For testing/debugging
    return {"text": "Hello World!"}

@app.post("/predict")
def predict(model_input: ModelInput, request: Request):
    '''
    View function handling the main /predict endpoint
    '''
    client_host = request.client.host
    inp = model_input.inputs
    logger.log(f'[{datetime.now()}] Received input of size {sys.getsizeof(inp)} from {client_host}')
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
    if isinstance(output, (np.ndarray, torch.Tensor)):
        output = output.tolist()
    return {"output": output}