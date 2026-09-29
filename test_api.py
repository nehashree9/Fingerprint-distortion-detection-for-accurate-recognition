import requests
import json
import sys

url = "http://127.0.0.1:8000/authenticate"
data = {"identity_code": "FAMILY-13/CHILD"}
files = {"fingerprint": open(r"data\test_distorted\distorted\FAMILY-13\CHILD\FM013_C1_distorted.png", "rb")}

try:
    response = requests.post(url, data=data, files=files)
    print("Status:", response.status_code)
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
except Exception as e:
    print("Error:", e)
