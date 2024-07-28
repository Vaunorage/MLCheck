import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.tree._tree import TREE_UNDEFINED
from z3 import *

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X, y)


def extract_and_convert_rules(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] if i != TREE_UNDEFINED else "undefined!" for i in tree.tree_.feature]
    value = tree.tree_.value

    constraints_list = []

    def recurse(left, right, threshold, features, node, constraints=[]):
        if threshold[node] != -2:  # Check if it's not a leaf node
            name = features[node]
            if left[node] != -1:  # Check if left child exists
                left_constraints = constraints + [Real(name) <= threshold[node]]
                recurse(left, right, threshold, features, left[node], left_constraints)
            if right[node] != -1:  # Check if right child exists
                right_constraints = constraints + [Real(name) > threshold[node]]
                recurse(left, right, threshold, features, right[node], right_constraints)
        else:
            constraints_list.append((constraints, np.argmax(value[node])))

    recurse(left, right, threshold, features, 0)
    return constraints_list


constraints_list = extract_and_convert_rules(clf, iris.feature_names)

s = Solver()

for constraints, classification in constraints_list:
    s.add(And(*constraints))

# Print out the solver
print(s)

# Optionally, check the satisfiability of the constraints
if s.check() == sat:
    print("The constraints are satisfiable.")
    print(s.model())
else:
    print("The constraints are not satisfiable.")
