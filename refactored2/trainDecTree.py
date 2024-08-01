
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from refactored2.util import local_save


def functrainDecTree():
    df = pd.read_csv('files/OracleData.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    model = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, min_samples_split=2, 
                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None)
    model = model.fit(X, Y)
    local_save(model, 'decTreeApprox', force_rewrite=True)
    # dump(model, 'files/Model/decTreeApprox.joblib')

    return model


