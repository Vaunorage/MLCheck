import json
import pickle

import numpy as np
import csv as cv
import pandas as pd
import os

from paths import HERE

files_folder = HERE.joinpath(f"refactored2/files")


def get_file_path(var_name, return_hypothetical=False, default_ext=None):
    for ext in ['.csv', '.json', '.pkl', '.txt']:
        var_path = files_folder.joinpath(f"{var_name}{ext}")
        if var_path.exists():
            return var_path

    # Return the hypothetical path if no file exists and return_hypothetical is True
    if return_hypothetical:
        return files_folder.joinpath(f"{var_name}{default_ext}")  # Placeholder for extension
    return None


def file_exists(var_name):
    for ext in ['.csv', '.json', '.pkl', '.txt']:
        var_path = files_folder.joinpath(f"{var_name}{ext}")
        if var_path.exists():
            return True
    return False


def local_delete(var_name):
    deleted_files = []
    for ext in ['.csv', '.json', '.pkl', '.txt']:
        var_path = files_folder.joinpath(f"{var_name}{ext}")
        if var_path.exists():
            var_path.unlink()  # Delete the file
            deleted_files.append(var_path)

    if not deleted_files:
        print(f"No files found to delete with base name '{var_name}'")

    return deleted_files


def local_save(var, var_name, force_rewrite=False):
    var_path = files_folder.joinpath(var_name)
    var_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    if isinstance(var, pd.DataFrame):
        var_path = var_path.with_suffix('.csv')
        if var_path.exists() and not force_rewrite:
            var.to_csv(var_path, mode='a', header=False, index=False)
        else:
            var.to_csv(var_path, index=False)
    elif isinstance(var, dict):
        var_path = var_path.with_suffix('.json')
        with open(var_path, 'w' if force_rewrite else 'a') as file:
            json.dump(var, file)
    elif isinstance(var, str):
        var_path = var_path.with_suffix('.txt')
        with open(var_path, 'w' if force_rewrite else 'a') as file:
            file.write(var + '\n')
    else:
        var_path = var_path.with_suffix('.pkl')
        if var_path.exists() and not force_rewrite:
            with open(var_path, 'rb') as file:
                existing_data = pickle.load(file)
            combined_data = existing_data + var if isinstance(existing_data, list) else [existing_data, var]
            with open(var_path, 'wb') as file:
                pickle.dump(combined_data, file)
        else:
            with open(var_path, 'wb') as file:
                pickle.dump(var, file)


def local_load(var_name):
    var_path_csv = files_folder.joinpath(f"{var_name}.csv")
    var_path_json = files_folder.joinpath(f"{var_name}.json")
    var_path_pkl = files_folder.joinpath(f"{var_name}.pkl")
    var_path_txt = files_folder.joinpath(f"{var_name}.txt")

    if var_path_json.exists():
        with open(var_path_json, 'r') as file:
            data = json.load(file)
    elif var_path_csv.exists():
        data = pd.read_csv(var_path_csv)
    elif var_path_pkl.exists():
        with open(var_path_pkl, 'rb') as file:
            data = pickle.load(file)
    elif var_path_txt.exists():
        with open(var_path_txt, 'r') as file:
            data = file.read()
    else:
        raise FileNotFoundError(f"No saved data found with base name '{var_name}'")

    return data


def run_z3(input_file, output_file):
    os.system(
        f"z3 {files_folder.joinpath(input_file).as_posix()}.txt > {files_folder.joinpath(output_file).as_posix()}.txt")


def convDataInst(X, df, j, no_of_class):
    paramDict = local_load('param_dict')
    if (paramDict['multi_label']):
        no_of_class = 1
    data_inst = np.zeros((1, df.shape[1] - no_of_class))
    if (j > X.shape[0]):
        raise Exception('Z3 has produced counter example with all 0 values of the features: Run the script Again')
    for i in range(df.shape[1] - no_of_class):
        data_inst[0][i] = X[j][i]
    return data_inst


def funcAdd2Oracle(data):
    local_save(pd.DataFrame(data), 'TestingData')


def funcCreateOracle(no_of_class, multi_label, model):
    df = local_load('TestingData')
    data = df.values
    if multi_label:
        X = data[:, :-1]
        X = dfTest.drop(self.paramDict['output_class_name'], axis=1)
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
    local_save(df, 'OracleData', force_rewrite=True)


def addSatOpt(file_name):
    hh = "\n (check-sat) \n (get-model)"
    local_save(hh, file_name)


def storeAssumeAssert(file_name, no_assumption=False):
    if not no_assumption:
        f2_content = local_load('assumeStmnt')
        local_save(f2_content, file_name)
    f3_content = local_load('assertStmnt')
    local_save(f3_content, file_name)


def update_dataframe_types(df: pd.DataFrame, categorical_cols=None) -> pd.DataFrame:

    def infer_and_convert(series, col_name):
        if categorical_cols and col_name in categorical_cols:
            return series.astype('category')
        try:
            return pd.to_numeric(series, errors='coerce')
        except ValueError:
            pass
        try:
            return pd.to_datetime(series, errors='coerce')
        except ValueError:
            pass
        if series.dropna().map(lambda x: isinstance(x, bool)).all():
            return series.astype(bool)
        return series.astype(str)

    return df.apply(lambda series: infer_and_convert(series, series.name))
