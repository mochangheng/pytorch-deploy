from typing import Callable, List, Dict, Union

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import sys
import requests

from .logger import Logger
from .simple_login import login
from getpass import getpass
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

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

def run_model(inp):
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
    
    return output

def create_logger(log_file: str) -> None:
    global logger
    logger = Logger(log_file)

@app.get("/")
def root():
    # For testing/debugging
    # Sending post request to login function in simple_login
    username = input("Enter your username: ")
    password = getpass("Enter your password: ")
    params = {'grant_type': '', 
              'username': username, 
              'password': password, 
              'scope': '', 
              'client_id': '', 
              'client_secret': ''}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    r = requests.post("http://127.0.0.1:8000/token", params=params, headers=headers)
    print(r.text)
    return {"text": "Hello World!"}

@app.post("/predict")
def predict(model_input: ModelInput, request: Request):
    '''
    View function handling the main /predict endpoint
    '''
    inp = model_input.inputs
    client_host = request.client.host
    logger.log(f'[{datetime.now()}] Received input of size {sys.getsizeof(inp)} from {client_host}')
    output = run_model(inp)
    return {"output": output}

@app.post("/predict_image")
def predict_file(file: UploadFile = File(...)):
    im = Image.open(file.file)
    inp = transforms.ToTensor()(im)
    inp = torch.unsqueeze(inp, 0)
    output = run_model(inp)
    return {"output": output}
