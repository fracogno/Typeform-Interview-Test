from flask import Flask, jsonify, make_response, request

from src.models import task_1 as models_t1
from src.utils import misc
app = Flask(__name__)

base_path = misc.get_base_path()


@app.route("/typeform/task_1", methods=['GET'])
def task_1():
    if "data" not in request.args:
        return make_response(jsonify({"error": "Missing data field."}), 400)

    data = request.args["data"]
    return jsonify({"prediction": data})
    ##
    #
    #model = models_t1.get_MLP()

    normalization_by_feature = misc.load_pickle(base_path + "pickles/task_1_max.py")
    #data /= normalization_by_feature

    #model.load_weights(base_path + "checkpoints/task_1/model")
    #return jsonify({"prediction": model.predict(data)})


if __name__ == "__main__":
    app.run(host='0.0.0.0')
