import requests
import json

url = ' http://127.0.0.1:5000/api/'

data = [[0.038076 , 0.050680 , 0.061696 , 0.021872, -0.044223 ,-0.034821, -0.043401, -0.002592,  0.019908, -0.017646]]
j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r.text)
