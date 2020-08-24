from typing import Callable, List, Dict, Union, Optional
import atexit
from collections.abc import Sequence
from copy import deepcopy
import os

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from datetime import datetime, timedelta
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
from jose import JWTError, jwt
from passlib.context import CryptContext

app = FastAPI()
config = None
inference_fn = None
pre = []
post = []
logger = None

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

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "3812c26372a49757aed42e302b37d2031cd546b649a1d7bf058191b29d99dcbe"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")



def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

    


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@app.get("/users/me/items/")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": "Foo", "owner": current_user.username}]

@app.get("/")
def root():
    # For testing/debugging

    return {"text": "Hello World!"}

@app.post("/predict")
def predict(model_input: ModelInput, request: Request):
    # Add access token checking code here
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
    im = Image.open(file.file)
    inp = transforms.ToTensor()(im)

    # Logging
    client_host = request.client.host
    logger.log(f'[{datetime.now()}] Received input of size {sys.getsizeof(inp)} from {client_host}')

    # Change the shape so it fits Conv2d
    inp = torch.unsqueeze(inp, 0)
    output = run_model(inp)
    return {"output": output}
