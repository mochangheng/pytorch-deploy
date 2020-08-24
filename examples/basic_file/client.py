import requests
import numpy as np

filename = "../palm.jpg"
files = {'file': open(filename, "rb")}
r = requests.post("http://127.0.0.1:8000/predict_image", files=files)
response = r.json()
output = np.array(response["output"])
print(np.argmax(output))