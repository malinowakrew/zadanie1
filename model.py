import pandas as pd
from os import path
from sklearn import preprocessing, model_selection, ensemble
import pickle


def model_iris(dataset_name="iris.csv"):
    data = pd.read_csv(path.join("./data", dataset_name))
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.20, shuffle=True)

    forest_clf = ensemble.RandomForestClassifier(random_state=0, n_estimators=50, max_depth=13)
    forest_clf.fit(X_train, y_train)

    filename = 'models/iris_random_forest_without_normalize.pickle'
    with open(filename, 'wb') as fp:
        pickle.dump(forest_clf, fp)


def model_iris_with_normalization(dataset_name="iris.csv"):
    data = pd.read_csv(path.join("./data", dataset_name))
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    x = X.values

    standard = preprocessing.Normalizer()
    x_scaled = standard.fit_transform(x)
    X = pd.DataFrame(x_scaled)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.20, shuffle=True)

    forest_clf = ensemble.RandomForestClassifier(random_state=0, n_estimators=50, max_depth=13)
    forest_clf.fit(X_train, y_train)

    filename = 'models/iris_random_forest_with_normalization.pickle'
    with open(filename, 'wb') as fp:
        pickle.dump(forest_clf, fp)


def model_golf(dataset_name="golf.csv"):
    data = pd.read_csv(path.join("./data", dataset_name), sep="\t")
    types = {}
    for nr, i in enumerate(data.columns):
        if data.iloc[0, nr] == "discrete":
            types[i] = "category"
        else:
            types[i] = "float"

    data = data.iloc[2:, :].astype(types)
    X = pd.get_dummies(data.iloc[:, :-1])
    Y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.20, shuffle=True)

    forest_clf = ensemble.RandomForestClassifier(random_state=0, n_estimators=40, max_depth=13)
    forest_clf.fit(X_train, y_train)

    filename = 'models/golf_random_forest_without_normalize.pickle'
    with open(filename, 'wb') as fp:
        pickle.dump(forest_clf, fp)

if __name__ == '__main__':
    model_iris()
    model_golf()
