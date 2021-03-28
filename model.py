import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from os import path
from sklearn import preprocessing, model_selection, ensemble, metrics
import pickle

def confusionMatrix(observed,predicted):
    cm = metrics.confusion_matrix(observed,predicted)/observed.shape[0]
    df_cm = pd.DataFrame(cm, index=["Actual True", "Actual False"], columns = ["Pred True", "Pred False"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)

def model(dataset_name):
    data = pd.read_csv(path.join("./data", dataset_name))
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    # x = X.values
    #
    # standard = preprocessing.StandardScaler()
    # x_scaled = standard.fit_transform(x)
    # X = pd.DataFrame(x_scaled)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.20, shuffle=True)

    forest_clf = ensemble.RandomForestClassifier(random_state=0, n_estimators=50, max_depth=13)
    forest_clf.fit(X_train, y_train)

    y_pred = forest_clf.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))
    #confusionMatrix(y_test, y_pred)

    filename = 'models/iris_random_forest_without_normalize.pickle'
    with open(filename, 'wb') as fp:
        pickle.dump(forest_clf, fp)


if __name__ == '__main__':
    model("iris.csv")
