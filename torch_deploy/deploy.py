from typing import Callable, List, Union, Optional
import importlib

import uvicorn
import torch.nn as nn

from .app import setup

def deploy(
    model: nn.Module,
    pre: Union[List[Callable], Callable] = None,
    post: Union[List[Callable], Callable] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    ssl_keyfile: str = None,
    ssl_certfile: str = None,
    ssl_ca_certs: str = None,
    logdir: str = "./deploy_logs/",
    inference_fn: str = None
) -> None:
    '''
    Main entrypoint of the library. This will start a FastAPI app which serves
    the model

    model: a PyTorch model which subclasses nn.Module and is callable
    pre: Function or list of functions which will be applied to the input
    post: Function or list of functions which will be applied to the output
    host: The address for serving the model
    port: The port for serving the model
    '''
    config = {
        "model": model,
        "pre": pre,
        "post": post,
        "logdir": logdir,
        "inference_fn": inference_fn
    }

    setup(config)

    kwargs = {
        "host": host,
        "port": port,
        "ssl_keyfile": ssl_keyfile,
        "ssl_certfile": ssl_certfile,
        "ssl_ca_certs": ssl_ca_certs
    }

    uvicorn.run("torch_deploy.app:app", **kwargs)
    