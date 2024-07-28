import shutil

import pandas as pd
import re
import numpy as np
from sklearn.tree import _tree


def tree_to_code(tree, feature_names, output_file):
    with open(output_file, 'w') as f:
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        f.write("def tree({}):\n".format(", ".join(feature_names)))

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                f.write("{}if {} <= {}:\n".format(indent, name, threshold))
                f.write("{}{{\n".format(indent))
                recurse(tree_.children_left[node], depth + 1)
                f.write("{}}}\n".format(indent))
                f.write("{}else:  # if {} > {}\n".format(indent, name, threshold))
                f.write("{}{{\n".format(indent))
                recurse(tree_.children_right[node], depth + 1)
                f.write("{}}}\n".format(indent))
            else:
                f.write("{}return {}\n".format(indent, np.argmax(tree_.value[node][0])))

        recurse(0, 1)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def funcConvBranch(single_branch, dfT, rep, smt_file):
    with open(smt_file, 'a') as f:
        f.write("(assert (=> (and ")
        for temp_Str in single_branch:
            if 'if' in temp_Str:
                fe_name = next(
                    (col for col in dfT.columns if col in temp_Str), None)
                if fe_name:
                    fe_index = dfT.columns.get_loc(fe_name)
                    data_type = str(dfT.dtypes[fe_index])
                    sign = re.search(r'[<>]=?', temp_Str).group()
                    digit = re.search(r'\d+\.?\d*', temp_Str).group()

                    if 'int' in data_type:
                        digit = int(float(digit))
                    else:
                        digit = float(digit)

                    f.write(f"({sign} {fe_name}{rep} {digit}) ")
            elif 'return' in temp_Str:
                digit_class = re.search(r'\d+', temp_Str).group()
                f.write(f") (= Class{rep} {digit_class})))\n")


def funcGetBranch(sinBranch, dfT, rep, smt_file):
    if any('return' in tempSt for tempSt in sinBranch):
        funcConvBranch(sinBranch, dfT, rep, smt_file)


def funcGenBranch(dfT, rep, tree_output_file, smt_file):
    with open(tree_output_file) as f1:
        file_content = [x.strip() for x in f1.readlines()]

    temp_file_cont = []
    for line in file_content[1:]:
        temp_file_cont.append(line)
        if line == '}':
            funcGetBranch(temp_file_cont, dfT, rep, smt_file)
            while temp_file_cont and temp_file_cont[-1] != '{':
                temp_file_cont.pop()
            if temp_file_cont:
                temp_file_cont.pop()

    if 'return' in file_content[1]:
        digit = re.search(r'\d+', file_content[1]).group()
        with open(smt_file, 'a') as f:
            f.write(f"(assert (= Class{rep} {digit}))\n")


def funcConv(dfT, no_of_instances, feName_type, tree_output_file, smt_file):
    with open(smt_file, 'w') as f:
        for j in range(no_of_instances):
            for col in dfT.columns[:-1]:
                fe_type = feName_type[col]
                decl = f"(declare-fun {col}{j} () {'Int' if 'int' in fe_type else 'Real'})"
                f.write(f"{decl}\n")
            f.write(f"; {j}th element\n")

        for i in range(no_of_instances):
            f.write(f"(declare-fun Class{i} () Int)\n")

        f.write('(define-fun absoluteInt ((x Int)) Int\n')
        f.write('  (ite (>= x 0) x (- x)))\n')

        f.write('(define-fun absoluteReal ((x Real)) Real\n')
        f.write('  (ite (>= x 0) x (- x)))\n')

    for i in range(no_of_instances):
        with open(smt_file, 'a') as f:
            f.write(f'\n;-----------{i}-----------number instance--------------\n')
        funcGenBranch(dfT, i, tree_output_file, smt_file)


def funcGenSMTFairness(df, no_of_instances, feName_type, tree_output_file, smt_file):
    funcConv(df, no_of_instances, feName_type, tree_output_file, smt_file)


def functree2LogicMain(tree, oracle_data_file_path, no_of_instances, output_class, tree_output_file,
                       smt_file, constraint_file):
    df = pd.read_csv(oracle_data_file_path)
    tree_to_code(tree, df.columns, tree_output_file)
    feName_type = {k: str(v) for k, v in dict(df.dtypes).items()}
    feName_type['Class'] = feName_type[output_class]
    funcGenSMTFairness(df, no_of_instances, feName_type, tree_output_file, smt_file)
    shutil.copyfile(smt_file, constraint_file)
