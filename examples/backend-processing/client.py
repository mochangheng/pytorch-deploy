import requests
from PIL import Image
import numpy as np
from torchvision import transforms

im = Image.open('palm.jpg')
body = {"inputs": np.asarray(im).tolist()}
r = requests.post("http://127.0.0.1:8000/predict", json=body)
response = r.json()
output = np.array(response["output"])
print(np.argmax(output.squeeze()))