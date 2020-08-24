import requests
from PIL import Image
import numpy as np
from torchvision import transforms

im = Image.open('../palm.jpg')
resize = transforms.Resize(224)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
tensor = normalize(to_tensor(resize(im))).unsqueeze(0)
body = {"inputs": tensor.tolist()}
r = requests.post("http://127.0.0.1:8000/predict", json=body)
response = r.json()
output = np.array(response["output"])
print(np.argmax(output.squeeze()))