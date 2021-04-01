from flask import Flask, request, jsonify

from os import path, environ
import pickle
import logging
from sklearn import metrics

from werkzeug.exceptions import InternalServerError

from error import AnyError
from model import model_iris, model_golf, basic_model_golf, model_iris_with_normalization
from load_dataset import load_dataset

model_iris()
model_golf()
basic_model_golf()
model_iris_with_normalization()


app = Flask(__name__)


@app.route('/hello')
def hello_world():
    return 'Niech moc będzie z rekrutującymi'


@app.route('/api/predictions', methods=['GET'])
def predictions():
    try:
        model_name = request.json.get('model_name', None)
        dataset_name = request.json.get('dataset_name', None)
        index_name = request.json.get('index_name', None)
        start_index = request.json.get('start_index', None)
        end_index = request.json.get('end_index', None)

        Y, X = load_dataset(dataset_name, start_index, end_index, index_name)

        with open(path.join("./models", model_name), 'rb') as fp:
            pickle_model = pickle.load(fp)

        y_pred = pickle_model.predict(X)

        return {"model accuracy": metrics.accuracy_score(Y, y_pred),
                "predictions": list(y_pred)}, 200

    except Exception as error:
        logging.error(error)
        raise AnyError


@app.errorhandler(AnyError)
def handle_bad_request(error):
    return jsonify({"route": False,
                    "text": str(error)}), 404


@app.errorhandler(InternalServerError)
def handle_internal_server_error(error):
    return jsonify({"route": False,
                    "text": str(error)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=int(environ.get('PORT', 8080)))
