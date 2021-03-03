from flask import Flask, jsonify, make_response, request
import numpy as np

from src.models import task_1 as models_t1
from src.utils import misc
app = Flask(__name__)

base_path = misc.get_base_path()
normalization_by_feature = misc.load_pickle(base_path + "pickles/task_1_max.py")


@app.route("/typeform/task_1", methods=['POST'])
def task_1():
    if "data" not in request.json:
        return make_response(jsonify({"error": "Missing data field."}), 400)

    data = np.array(request.json["data"])
    if len(data.shape) != 2 or data.shape[1] != 47:
        return make_response(jsonify({"error": "Wrong data shape."}), 400)

    model = models_t1.get_MLP()
    model.load_weights(base_path + "checkpoints/task_1/model")
    data /= normalization_by_feature
    return jsonify({"prediction": str(model.predict(data))})


if __name__ == "__main__":
    app.run(host='0.0.0.0')
