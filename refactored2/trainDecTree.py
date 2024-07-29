
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from joblib import dump


def functrainDecTree():
    df = pd.read_csv('files/OracleData.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    model = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, min_samples_split=2, 
                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None)
    model = model.fit(X, Y)
    dump(model, 'files/Model/decTreeApprox.joblib')

    return model


