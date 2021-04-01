import pandas as pd
from sklearn import preprocessing, model_selection, ensemble, linear_model
import pickle

from load_dataset import load_dataset


def model_iris(dataset_name="iris.csv"):
    Y, X = load_dataset(dataset_name, 0, -1, "species")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.20, shuffle=True)

    forest_clf = ensemble.RandomForestClassifier(random_state=0, n_estimators=50, max_depth=13)
    forest_clf.fit(X_train, y_train)

    filename = "models/iris_random_forest_without_normalization.pickle"
    with open(filename, 'wb') as fp:
        pickle.dump(forest_clf, fp)


# I don't normalize loaded data in entrypoint so this score must be low
# it has only presentation value
def model_iris_with_normalization(dataset_name="iris.csv"):
    Y, X = load_dataset(dataset_name, 0, -1, "species")
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
    Y, X = load_dataset(dataset_name, 0, -1, "Play")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.20, shuffle=True)

    forest_clf = ensemble.RandomForestClassifier(random_state=0, n_estimators=40, max_depth=13)
    forest_clf.fit(X_train, y_train)

    filename = 'models/golf_random_forest_without_normalization.pickle'
    with open(filename, 'wb') as fp:
        pickle.dump(forest_clf, fp)


def basic_model_golf(dataset_name="golf.csv"):
    Y, X = load_dataset(dataset_name, 0, -1, "Play")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.20, shuffle=True)

    forest_clf = linear_model.LogisticRegression()
    forest_clf.fit(X_train, y_train)

    filename = 'models/golf_logistic_regression.pickle'
    with open(filename, 'wb') as fp:
        pickle.dump(forest_clf, fp)


if __name__ == '__main__':
    model_iris()
    model_golf()
