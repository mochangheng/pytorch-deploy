# torch-deploy

## Installation
To install:
```
pip install pytorch-deploy
```

You also have to install `torch` and `torchvision`. You can do so [here](https://pytorch.org/get-started/locally/).

## Usage
Deploying a pretrained ResNet-18:
```python
import torch
import torchvision.models as models
from torch_deploy import deploy

resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
deploy(resnet18, pre=torch.tensor)
```

The default host and port is 0.0.0.0:8000.

## Endpoints

You can access the docs for the endpoints at "host:port/docs" after running `python server.py`.

### /predict
Request body: application/json <br>
Response body: application/json

Here's an example of how to use to use the /predict endpoint.

```python
import requests
from PIL import Image
import numpy as np
from torchvision import transforms

im = Image.open('palm.jpg')
resize = transforms.Resize(224)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
tensor = normalize(to_tensor(resize(im))).unsqueeze(0)
body = {"inputs": tensor.tolist()}
r = requests.post("http://127.0.0.1:8000/predict", json=body)
response = r.json()
output = np.array(response["output"])
```

**Note** that you need to send the model input in the request JSON body under the field "inputs".
If you want to send a tensor or a numpy array in the request, you need to turn it into a list first.

The output of the model will be in the response JSON body under the "output" field.

Sample response format:
```python
response = {"output": (your numpy array as a list here)}
```

### /predict_image
Request body: multipart/form-data <br>
Response body: application/json

Here's an example of how to use to use the /predict_image endpoint.

```python
import requests
import numpy as np

filename = "../palm.jpg"
files = {'file': open(filename, "rb")}
r = requests.post("http://127.0.0.1:8000/predict_image", files=files)
response = r.json()
output = np.array(response["output"])
print(np.argmax(output))
```

The file is uploaded with the content type "multipart/form-data". This requires minimal work on the client side and is compatible with standard file upload requests.

## Documentation
```python
torch_deploy.deploy(
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
)
```

Easily converts a pytorch model to API for production usage.

- `model`: A PyTorch model which subclasses nn.Module and is callable. Model used for the API.
- `pre`: A function or list of functions to be applied to the input.
- `post`: Function or list of functions applied to model output before being sent as a response.
- `host`: The address for serving the model.
- `port`: The port for serving the model.
- `ssl_keyfile`, `ssl_certfile`, `ssl_ca_certs`: SSL configurations that are passed to uvicorn
- `logfile`: Filename to create a file that stores date, ip address, and size of input for each access of the API. If `None`, no file will be created.
- `inference_fn`: Name of the method of the model that should be called for the inputs. If `None`, the model itself will be called (If `model` is a `nn.Module` then it's equivalent to calling `model.forward(inputs)`).

## Examples
There are some sample code in the examples/ directory.

## Currently In Progress
Still working on an OAuth2 login system that requires correct user credentials to use torch-deploy.

## Dependencies
`torch, torchvision, fastapi, uvicorn, requests, numpy, pydantic`
