import requests
import numpy as np

data = [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 1.0, 4.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 11.0, 7.0, 21.0]]
pred = np.array(requests.post("http://188.166.213.241:5000/typeform/task_1", json={"data": data}).json()["prediction"])
print(pred)
