from typing import Callable, List, Dict, Union
import atexit
from collections.abc import Sequence
from copy import deepcopy
import os

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import torch
import sys

from .logger import Logger

app = FastAPI(
    title="torch-deploy",
    description="one line deployment for pytorch models"
)
config = None
inference_fn = None
pre = []
post = []
logger = None

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
@atexit.register
def cleanup():
    if logger is not None:
        logger.close()


class ModelInput(BaseModel):
    '''Pydantic Model to receive parameters for the /predict endpoint'''
    inputs: Union[List, Dict]

def setup(my_config):
    '''Initialize the global variables'''
    global inference_fn, pre, post, logger, config
    config = deepcopy(my_config)

    # Make log directory if it doesn't exist
    my_logdir = config["logdir"]
    if not os.path.isdir(my_logdir):
        os.mkdir(my_logdir)

    # Init logger
    logger = Logger(os.path.join(my_logdir, "logfile"))

    # Init inference_fn
    model = config["model"]
    if config["inference_fn"] is not None:
        inference_fn = getattr(model, config["inference_fn"])
    else:
        inference_fn = model

    # Init preprocessing and postprocessing functions
    my_pre = config["pre"]
    my_post = config["post"]
    if my_pre:
        if isinstance(my_pre, Sequence):
            pre = list(my_pre)
        else:
            pre = [my_pre]
    if my_post:
        if isinstance(my_post, Sequence):
            post = list(my_post)
        else:
            post = [my_post]


def run_model(inp):
    # Apply all preprocessing functions
    for f in pre:
        inp = f(inp)
    
    # Pass input through model
    output = inference_fn(inp)

    # Apply all postprocessing functions
    for f in post:
        output = f(output)
    
    # If torch tensor or numpy array, transform to list so we can pass it back
    if isinstance(output, (np.ndarray, torch.Tensor)):
        output = output.tolist()
    
    return output

@app.get("/")
def root():
    # For testing/debugging
    return {"text": "Hello World!"}

@app.post("/predict")
def predict(model_input: ModelInput, request: Request):
    '''
    View function handling the main /predict endpoint
    Input: Expect to receive an application/json body. The value of the "inputs" field
           will be used as the input that will be passed to the model 
           and should be a list or a dict.
    Output: The output of the model after being run through the postprocessing
            functions.
    '''
    inp = model_input.inputs

    # Logging
    client_host = request.client.host
    logger.log(f'[{datetime.now()}] Received input of size {sys.getsizeof(inp)} from {client_host}')

    output = run_model(inp)
    return {"output": output}

@app.get("/predict_image")
def upload_image(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict_image")
def predict_image(request: Request, file: UploadFile = File(...)):
    '''
    View function handling the /predict_image endpoint
    Input: Expect to receive a  body. The value of the "inputs" field
           will be used as the input that will be passed to the model 
           and should be a list or a dict.
    Output: The output of the model after being run through the postprocessing
            functions.
    '''
    inp = Image.open(file.file)

    # Logging
    client_host = request.client.host
    logger.log(f'[{datetime.now()}] Received input of size {sys.getsizeof(inp)} from {client_host}')

    output = run_model(inp)
    return {"output": output}
        