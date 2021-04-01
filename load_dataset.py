import pandas as pd
from os import path
from error import AnyError


def load_dataset(dataset_name, start_index, end_index, index_name):
    try:
        if dataset_name == "iris.csv":
            data = pd.read_csv(path.join("./data", dataset_name)).iloc[start_index:end_index, :]
            Y = data[index_name]
            X = data.drop(index_name, axis=1)

        else:
            data = pd.read_csv(path.join("./data", dataset_name), sep="\t")
            types = {}
            for nr, i in enumerate(data.columns):
                if data.iloc[0, nr] == "discrete":
                    types[i] = "category"
                else:
                    types[i] = "float"

            data = data.iloc[2:, :].astype(types).iloc[start_index:end_index, :]
            Y = data[index_name]
            X = pd.get_dummies(data.drop(index_name, axis=1))

        return Y, X

    except Exception as any_error:
        raise AnyError