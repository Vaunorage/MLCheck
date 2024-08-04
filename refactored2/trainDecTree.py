import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from refactored2.util import local_save, local_load


def train_decision_tree():
    df = local_load('OracleData')
    X = df.drop(columns='Class').values
    y = df['Class'].values

    model = DecisionTreeClassifier(
        criterion="entropy",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None
    )

    model.fit(X, y)
    local_save(model, 'decTreeApprox', force_rewrite=True)

    return model
