import pandas as pd
import csv as cv
import numpy as np

from refactored2.util import local_load

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import re

from sklearn.tree import _tree


def tree_to_code(tree, feature_names):
    f = open('files/TreeOutput.txt', 'w')
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    f.write("def tree({}):".format(", ".join(feature_names)))
    f.write("\n")

    def recurse(node, depth):
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            f.write("{}if {} <= {}:".format(indent, name, threshold))
            f.write("\n")

            f.write("{}".format(indent) + "{")
            f.write("\n")

            recurse(tree_.children_left[node], depth + 1)

            f.write("{}".format(indent) + "}")
            f.write("\n")

            f.write("{}else:  # if {} > {}".format(indent, name, threshold))
            f.write("\n")

            f.write("{}".format(indent) + "{")
            f.write("\n")

            recurse(tree_.children_right[node], depth + 1)

            f.write("{}".format(indent) + "}")
            f.write("\n")

        else:
            f.write("{}return {}".format(indent, np.argmax(tree_.value[node][0])))
            f.write("\n")

    recurse(0, 1)
    f.close()


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def funcConvBranch(single_branch, dfT, rep):
    f = open('files/DecSmt.smt2', 'a')
    f.write("(assert (=> (and ")
    for i in range(0, len(single_branch)):
        temp_Str = single_branch[i]
        if ('if' in temp_Str):
            # temp_content[i] = content[i]
            for j in range(0, dfT.columns.values.shape[0]):
                if (dfT.columns.values[j] in temp_Str):
                    fe_name = str(dfT.columns.values[j])
                    fe_index = j

            data_type = str(dfT.dtypes[fe_index])

            if ('<=' in temp_Str):
                sign = '<='
            elif ('<=' in temp_Str):
                sign = '>'
            elif ('>' in temp_Str):
                sign = '>'
            elif ('>=' in temp_Str):
                sign = '>='

            if ('int' in data_type):
                digit = int(re.search(r'\d+', temp_Str).group(0))
            elif ('float' in data_type):
                digit = float(re.search(r'\d+', temp_Str).group(0))
            digit = str(digit)
            f.write("(" + sign + " " + fe_name + str(rep) + " " + digit + ") ")

        elif ('return' in temp_Str):
            digit_class = int(re.search(r'\d+', temp_Str).group(0))
            digit_class = str(digit_class)
            f.write(") (= Class" + str(rep) + " " + digit_class + ")))")
            f.write('\n')
    f.close()


def funcGetBranch(sinBranch, dfT, rep):
    for i in range(0, len(sinBranch)):
        tempSt = sinBranch[i]
        if ('return' in tempSt):
            funcConvBranch(sinBranch, dfT, rep)


def funcGenBranch(dfT, rep):
    with open('files/TreeOutput.txt') as f1:
        file_content = f1.readlines()
    file_content = [x.strip() for x in file_content]

    f1.close()

    noOfLines = file_len('files/TreeOutput.txt')
    temp_file_cont = ["" for x in range(noOfLines)]

    i = 1
    k = 0
    while (i < noOfLines):

        j = k - 1
        if temp_file_cont[j] == '}':
            funcGetBranch(temp_file_cont, dfT, rep)
            while True:
                if (temp_file_cont[j] == '{'):
                    temp_file_cont[j] = ''
                    temp_file_cont[j - 1] = ''
                    j = j - 1
                    break
                elif (j >= 0):
                    # print(temp_file_cont.pop(i))
                    temp_file_cont[j] = ''
                    j = j - 1

            k = j

        else:
            temp_file_cont[k] = file_content[i]
            # print(temp_file_cont)
            k = k + 1
            i = i + 1
            # print(temp_file_cont.shape)

    # return temp_file_cont
    # print(temp_file_cont)

    if ('return' in file_content[1]):
        digit = int(re.search(r'\d+', file_content[1]).group(0))
        f = open('files/DecSmt.smt2', 'a')
        f.write("(assert (= Class" + str(rep) + " " + str(digit) + "))")
        f.write("\n")
        f.close()
    else:
        funcGetBranch(temp_file_cont, dfT, rep)


def funcConv(dfT, no_of_instances):
    # s = Stack()

    temp_content = ['']
    rep = 0
    min_val = 0
    max_val = 0

    # with open('files/feNameType.csv') as csv_file:
    #     reader = cv.reader(csv_file)
    #     feName_type = dict(reader)

    feName_type = local_load('feNameType')

    with open('files/TreeOutput.txt') as f1:
        content = f1.readlines()

    f = open('files/DecSmt.smt2', 'w')
    for j in range(0, no_of_instances):
        for i in range(0, dfT.columns.values.shape[0] - 1):
            tempStr = dfT.columns.values[i]
            ''' 
            fe_type = dfT.dtypes[i]
            fe_type = str(fe_type)
            '''
            fe_type = feName_type[tempStr]

            min_val = dfT.iloc[:, i].min()
            max_val = dfT.iloc[:, i].max()

            if ('int' in fe_type):
                f.write("(declare-fun " + tempStr + str(j) + " () Int)")
                f.write('\n')
                f.write('\n')
            elif ('float' in fe_type):
                f.write("(declare-fun " + tempStr + str(j) + " () Real)")
                f.write('\n')
                f.write('\n')
        f.write("; " + str(j) + "th element")
        f.write('\n')

    for i in range(0, no_of_instances):
        f.write("(declare-fun Class" + str(i) + " () Int)")
        f.write('\n')

    # Writing the functions for computing absolute integer & real value
    f.write('(define-fun absoluteInt ((x Int)) Int \n')
    f.write('  (ite (>= x 0) x (- x))) \n')

    f.write('(define-fun absoluteReal ((x Real)) Real \n')
    f.write('  (ite (>= x 0) x (- x))) \n')
    f.close()

    # Calling function to get the branch and convert it to z3 form,  creating alias
    for i in range(0, no_of_instances):
        f = open('files/DecSmt.smt2', 'a')
        f.write('\n;-----------' + str(i) + '-----------number instance-------------- \n')
        f.close()
        funcGenBranch(dfT, i)


def funcGenSMTFairness(df, no_of_instances):
    funcConv(df, no_of_instances)


def functree2LogicMain(tree, no_of_instances):
    df = pd.read_csv('files/OracleData.csv')
    tree_to_code(tree, df.columns)
    funcGenSMTFairness(df, no_of_instances)
