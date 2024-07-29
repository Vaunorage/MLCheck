import pickle

import numpy as np
import csv as cv
import pandas as pd
import os

from paths import HERE


def local_save(var, var_name):
    var_path = HERE.joinpath(f"refactored2/files/{var_name}")
    var_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    if isinstance(var, pd.DataFrame):
        var_path = var_path.with_suffix('.csv')
        var.to_csv(var_path, index=False)
    elif isinstance(var, dict):
        var_path = var_path.with_suffix('.json')
        with open(var_path, 'w') as file:
            json.dump(var, file)
    else:
        var_path = var_path.with_suffix('.pkl')
        with open(var_path, 'wb') as file:
            pickle.dump(var, file)
    print(f"Data has been saved to '{var_path}'")


def local_load(var_name):
    var_path_csv = HERE.joinpath(f"refactored2/files/{var_name}.csv")
    var_path_json = HERE.joinpath(f"refactored2/files/{var_name}.json")
    var_path_pkl = HERE.joinpath(f"refactored2/files/{var_name}.pkl")

    if var_path_csv.exists():
        data = pd.read_csv(var_path_csv)
        print(f"Data loaded as pandas DataFrame from '{var_path_csv}'")
    elif var_path_json.exists():
        with open(var_path_json, 'r') as file:
            data = json.load(file)
        print(f"Data loaded as dictionary from '{var_path_json}'")
    elif var_path_pkl.exists():
        with open(var_path_pkl, 'rb') as file:
            data = pickle.load(file)
        print(f"Data loaded using pickle from '{var_path_pkl}'")
    else:
        raise FileNotFoundError(f"No saved data found with base name '{var_name}'")

    return data


def file_len(fname):
    if (os.stat(fname).st_size == 0):
        return 'empty'
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def convDataInst(X, df, j, no_of_class):
    with open('files/param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)
    if (paramDict['multi_label'] == 'False'):
        no_of_class = 1
    data_inst = np.zeros((1, df.shape[1] - no_of_class))
    if (j > X.shape[0]):
        raise Exception('Z3 has produced counter example with all 0 values of the features: Run the script Again')
    for i in range(df.shape[1] - no_of_class):
        data_inst[0][i] = X[j][i]
    return data_inst


def funcAdd2Oracle(data):
    with open('files/TestingData.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(data)


def funcCreateOracle(no_of_class, multi_label, model):
    df = pd.read_csv('files/TestingData.csv')
    data = df.values
    if multi_label == 'False':
        X = data[:, :-1]
        predict_class = model.predict(X)
        for i in range(0, X.shape[0]):
            df.loc[i, 'Class'] = predict_class[i]
    else:
        X = data[:, :-no_of_class]
        predict_class = model.predict(X)
        index = df.shape[1] - no_of_class
        for i in range(0, no_of_class):
            className = str(df.columns.values[index + i])
            for j in range(0, X.shape[0]):
                df.loc[j, className] = predict_class[j][i]
    df.to_csv('files/OracleData.csv', index=False, header=True)


def addContent(file_name, f_content):
    f1 = open(file_name, 'a')
    for x in f_content:
        f1.write('\n')
        f1.write(x)
    f1.close()


def addSatOpt(file_name):
    f = open(file_name, 'a')
    f.write('\n')
    f.write("(check-sat) \n")
    f.write("(get-model) \n")


def storeAssumeAssert(file_name, no_assumption=False):
    if not no_assumption:
        with open('files/assumeStmnt.txt') as f2:
            f2_content = f2.readlines()
        f2_content = list(set([x.strip() for x in f2_content]))
        addContent(file_name, f2_content)
    with open('files/assertStmnt.txt') as f3:
        f3_content = f3.readlines()
    f3_content = list(set([x.strip() for x in f3_content]))
    addContent(file_name, f3_content)
