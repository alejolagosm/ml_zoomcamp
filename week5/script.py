"""
# First, use pickle to read and use the model directly

import pickle

with open("dv.bin", "rb") as f_in:
    dv = pickle.load(f_in)

with open("model1.bin", "rb") as f_in:
    model = pickle.load(f_in)

client = {
    "reports": 0,
    "share": 0.001694,
    "expenditure": 0.12,
    "owner": "yes",
}

X = dv.transform([client])

print(model.predict_proba(X)[0,1])
"""
# Additional script to use the service.py file that create a flask webservice

import requests

url = 'http://localhost:9696/predict'
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}

print(requests.post(url, json=client).json())