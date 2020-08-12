import requests
import numpy as np

array = np.zeros((1, 3, 256, 256)).tolist()
body = {"inputs": {"array": array}}
r = requests.post("http://127.0.0.1:8000/predict", json=body)
response = r.json()
output = np.array(response["output"])
print(output.shape)