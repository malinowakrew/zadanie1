from flask import Flask, request, jsonify
import pandas as pd
from os import path, environ
import pickle

from sklearn import metrics

from error import AnyError

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Niech moc będzie z rekrutującymi'


@app.route('/api/predictions', methods=['GET'])
def predictions():
    model_name = request.json.get('model_name', None)
    dataset_name = request.json.get('dataset_name', None)
    index_name = request.json.get('index_name', None)
    start_index = request.json.get('start_index', None)
    end_index = request.json.get('end_index', None)

    data = load_dataset(dataset_name, start_index, end_index, index_name)

    Y = data[index_name]
    X = data.drop(index_name, axis=1)

    with open(path.join("./models", model_name), 'rb') as fp:
        pickle_model = pickle.load(fp)

    y_pred = pickle_model.predict(X)

    return {"model accuracy": metrics.accuracy_score(Y, y_pred)}, 200


def load_dataset(dataset_name, start_index, end_index, index_name):
    try:
        return pd.read_csv(path.join("./data", dataset_name)).iloc[start_index:end_index, :]
        #.set_index(index_name)
        # yield pd.read_csv(path.join("./data", dataset_name)).set_index(index_name).iloc[start_index:end_index, :].Index

    except Exception as any_error:
        raise AnyError


@app.errorhandler(AnyError)
def handle_bad_request(error):
    return jsonify({"route": False,
                    "text": str(error)}), 404


if __name__ == '__main__':
    app.run(debug=True, port=int(environ.get('PORT', 8080)))
