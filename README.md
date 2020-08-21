# pytorch-deploy

## Usage
```
from torch_deploy import deploy
deploy(your_model)
```

## deploy Function
`deploy(model: nn.Module,
    pre: Union[List[Callable], Callable] = None,
    post: Union[List[Callable], Callable] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    logfile: str = None)`

Easily converts a pytorch model to API for production usage.

- `model`: A PyTorch model which subclasses nn.Module and is callable. Model used for the API.
- `pre`: A function or list of functions to be applied to the input.
- `post`: Function or list of functions applied to model output before being sent as a response.
- `host`: The address for serving the model.
- `port`: The port for serving the model.
- `logfile`: filename to create a file that stores date, ip address, and size of input for each access of the API. If `None`, no file will be created.

## Sample Response Format

## Sample Code

## Testing
Run `python test_server.py` first and then `python test_client.py` in another window to test.

## Dependencies
`torch, torchvision, fastapi[all], requests, numpy, pydantic`
