from flask import Flask, jsonify, make_response, request

app = Flask(__name__)


@routes.route("/typeform/task_1", methods=['GET'])
def task_1():
    ##return make_response(jsonify({"error": message}), 400)
    #request.args["name"]
    return jsonify({"prediction": 1})


if __name__ == "__main__":
    app.run(host='0.0.0.0')
