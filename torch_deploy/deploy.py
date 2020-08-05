from typing import Callable, List, Union
from collections.abc import Sequence
import importlib

import uvicorn
import torch.nn as nn

from .app import register_model, register_pre, register_post

def deploy(
    model: nn.Module,
    pre: Union[List[Callable], Callable] = None,
    post: Union[List[Callable], Callable] = None,
    host: str = "0.0.0.0",
    port: int = 8000
) -> None:
    register_model(model)
    if pre:
        if isinstance(pre, Sequence):
            register_pre(list(pre))
        else:
            register_pre([pre])
    if post:
        if isinstance(post, Sequence):
            register_post(list(post))
        else:
            register_post([post])

    uvicorn.run("torch_deploy.app:app", host=host, port=port)
    