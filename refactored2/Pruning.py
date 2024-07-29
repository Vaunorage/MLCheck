import pandas as pd
import csv as cv
import numpy as np
from refactored2 import util
import os
from refactored2 import ReadZ3Output
from sklearn.tree import _tree

from refactored2.util import local_load


def getDataType(value, dfOrig, i):
    data_type = str(dfOrig.dtypes[i])
    if ('int' in data_type):
        digit = int(value)
    elif ('float' in data_type):
        digit = float(value)
    return digit


def funcAddCond2File(index):
    temp_cond_content = ''
    with open('files/ConditionFile.txt') as fileCond:
        condition_file_content = fileCond.readlines()
    condition_file_content = [x.strip() for x in condition_file_content]

    with open('files/DecSmt.smt2') as fileSmt:
        smt_file_content = fileSmt.readlines()

    smt_file_content = [x.strip() for x in smt_file_content]
    smt_file_lines = util.file_len('files/DecSmt.smt2')
    fileCondSmt = open('files/ToggleBranchSmt.smt2', 'w')

    for i in range(smt_file_lines):
        fileCondSmt.write(smt_file_content[i])
        fileCondSmt.write("\n")
    fileCondSmt.close()

    with open('files/ToggleBranchSmt.smt2', 'r') as fileCondSmt:
        text = fileCondSmt.read()
        text = text.replace("(check-sat)", '')
        text = text.replace("(get-model)", '')

        with open('files/ToggleBranchSmt.smt2', 'w') as fileCondSmt:
            fileCondSmt.write(text)

    fileCondSmt = open('files/ToggleBranchSmt.smt2', 'a')

    temp_cond_content = condition_file_content[index]

    fileCondSmt.write("(assert (not " + temp_cond_content + "))")
    fileCondSmt.write("\n")

    fileCondSmt.write("(check-sat) \n")
    fileCondSmt.write("(get-model) \n")

    fileCondSmt.close()


def funcWrite2File(file_name):
    with open(file_name) as fileSmt:
        smt_file_content = fileSmt.readlines()

    smt_file_content = [x.strip() for x in smt_file_content]

    smt_file_lines = util.file_len(file_name)

    fileTogFeSmt = open('files/ToggleFeatureSmt.smt2', 'w')

    for i in range(smt_file_lines):
        fileTogFeSmt.write(smt_file_content[i])
        fileTogFeSmt.write("\n")

    fileTogFeSmt.close()


def funcgetPath4multiLbl(tree, dfMain, noCex, no_param):
    feature_names = dfMain.columns
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    dfT = pd.read_csv('files/TestDataSMTMain.csv')
    pred_arr = np.zeros((tree_.n_outputs))

    node = 0
    depth = 1
    f1 = open('files/SampleFile.txt', 'w')
    f1.write("(assert (=> (and ")
    pathCondFile = open('files/ConditionFile.txt', 'w')

    while (True):
        name = feature_name[node]
        threshold = tree_.threshold[node]

        for i in range(0, dfT.shape[1]):
            if (dfT.columns.values[i] == name):
                index = i

        if (tree_.feature[node] == _tree.TREE_UNDEFINED):
            for i in range(0, tree_.n_outputs):
                pred_arr[i] = np.argmax(tree_.value[node][i])
            if (no_param == 1):
                f1.write(") (= " + str(pred_arr) + ")))")
            else:
                f1.write(") (= " + str(pred_arr) + " " + str(noCex) + ")))")
            break

        index = int(index)

        if (dfT.iloc[0][index] <= threshold):

            node = tree_.children_left[node]

            depth = depth + 1

            threshold = getDataType(threshold, dfMain, index)
            threshold = round(threshold, 5)
            if (no_param == 1):
                f1.write("(<= " + str(name) + " " + str(threshold) + ") ")
            else:
                f1.write("(<= " + str(name) + " " + str(noCex) + " " + str(threshold) + ") ")
            pathCondFile.write("(<= " + str(name) + " " + str(threshold) + ") ")
            pathCondFile.write("\n")



        else:

            node = tree_.children_right[node]

            depth = depth + 1

            threshold = getDataType(threshold, dfMain, index)
            threshold = round(threshold, 5)
            # print(threshold)
            if (no_param == 1):
                f1.write("(> " + str(name) + " " + str(threshold) + ") ")
            else:
                f1.write("(> " + str(name) + " " + str(noCex) + " " + str(threshold) + ") ")

            pathCondFile.write("(> " + str(name) + " " + str(threshold) + ") ")
            pathCondFile.write("\n")

    f1.close()
    pathCondFile.close()


def funcgetPath(tree, dfMain, noCex):
    feature_names = dfMain.columns
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    dfT = pd.read_csv('files/TestDataSMTMain.csv')

    node = 0
    depth = 1
    f1 = open('files/SampleFile.txt', 'w')
    f1.write("(assert (=> (and ")
    pathCondFile = open('files/ConditionFile.txt', 'w')

    while (True):
        name = feature_name[node]
        threshold = tree_.threshold[node]

        for i in range(0, dfT.shape[1]):
            if (dfT.columns.values[i] == name):
                index = i

        if (tree_.feature[node] == _tree.TREE_UNDEFINED):
            f1.write(") (= Class " + str(np.argmax(tree_.value[node][0])) + ")))")
            break

        index = int(index)
        # print(dfT.iloc[noCex][index])
        if (dfT.iloc[noCex][index] <= threshold):
            node = tree_.children_left[node]
            depth = depth + 1
            threshold = getDataType(threshold, dfMain, index)
            f1.write("(<= " + str(name) + str(noCex) + " " + str(threshold) + ") ")
            pathCondFile.write("(<= " + str(name) + str(noCex) + " " + str(threshold) + ") ")
            pathCondFile.write("\n")
        else:
            node = tree_.children_right[node]
            depth = depth + 1
            threshold = getDataType(threshold, dfMain, index)
            f1.write("(> " + str(name) + str(noCex) + " " + str(threshold) + ") ")
            pathCondFile.write("(> " + str(name) + str(noCex) + " " + str(threshold) + ") ")
            pathCondFile.write("\n")

    f1.close()
    pathCondFile.close()


def funcPrunInst(dfOrig, dnn_flag):
    # data set to hold set of candidate counter examples, refer to cand-set of prunInst algorithm
    with open('files/CandidateSetInst.csv', 'w', newline='') as csvfile:
        fieldnames = dfOrig.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)

    paramDict = local_load('param_dict')

    if (paramDict['multi_label']):
        noClass = int(paramDict['no_of_class'])
    else:
        noClass = 1

    # Getting the counter example pair (x, x') and saving it to a permanent storage
    dfRead = pd.read_csv('files/TestDataSMTMain.csv')
    dataRead = dfRead.values

    # Combining loop in line 2 & 6 in a single loop
    for j in range(0, dfRead.shape[0]):
        for i in range(0, dfRead.columns.values.shape[0] - noClass):
            # writing content of files/DecSmt.smt2 to another file named files/ToggleFeatureSmt.smt2
            if (dnn_flag == True):
                funcWrite2File('DNNSmt.smt2')
            else:
                funcWrite2File('files/DecSmt.smt2')

            with open('files/ToggleFeatureSmt.smt2', 'r') as file:
                text = file.read()
                text = text.replace("(check-sat)", '')
                text = text.replace("(get-model)", '')
                with open('files/ToggleFeatureSmt.smt2', 'w') as file:
                    file.write(text)

            fileTogFe = open('files/ToggleFeatureSmt.smt2', 'a')
            name = str(dfRead.columns.values[i])

            data_type = str(dfOrig.dtypes[i])
            if ('int' in data_type):
                digit = int(dataRead[j][i])
            elif ('float' in data_type):
                digit = float(dataRead[j][i])

            digit = str(digit)
            if 'e' in digit:
                dig = digit.split('e')
                digit = dig[0]
            if ((int(paramDict['no_of_params']) == 1) and (paramDict['multi_label']) and (
                    paramDict['white_box_model'] == 'Decision tree')):
                fileTogFe.write("(assert (not (= " + name + " " + digit + "))) \n")
            else:
                fileTogFe.write("(assert (not (= " + name + str(j) + " " + digit + "))) \n")
            fileTogFe.write("(check-sat) \n")
            fileTogFe.write("(get-model) \n")
            fileTogFe.close()

            os.system(r"z3 files/ToggleFeatureSmt.smt2 > files/FinalOutput.txt")

            satFlag = ReadZ3Output.funcConvZ3OutToData(dfOrig)

            # If sat then add the counter example to the candidate set, refer line 8,9 in prunInst algorithm
            if (satFlag == True):
                dfSmt = pd.read_csv('files/TestDataSMT.csv')
                dataAppend = dfSmt.values
                with open('files/CandidateSetInst.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(dataAppend)


def funcPrunBranch(dfOrig, tree_model):
    noPathCond = 0
    # data set to hold set of candidate counter examples, refer to cand-set of prunBranch algorithm
    with open('files/CandidateSetBranch.csv', 'w', newline='') as csvfile:
        fieldnames = dfOrig.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)


    paramDict = local_load('param_dict')

    dfRead = pd.read_csv('files/TestDataSMTMain.csv')

    for row in range(0, dfRead.shape[0]):
        if (paramDict['multi_label']):
            funcgetPath4multiLbl(tree_model, dfOrig, row, int(paramDict['no_of_params']))
        else:
            funcgetPath(tree_model, dfOrig, row)
        fileCond = open('files/TreeOutput.txt', 'r')
        first = fileCond.read(1)

        if not first:
            print('No Branch')
        else:
            noPathCond = util.file_len('files/ConditionFile.txt')
            if (noPathCond == 'empty'):
                return
            for i in range(noPathCond):
                funcAddCond2File(i)
                os.system(r"z3 files/ToggleBranchSmt.smt2 > files/FinalOutput.txt")
                satFlag = ReadZ3Output.funcConvZ3OutToData(dfOrig)

                if (satFlag == True):
                    dfSmt = pd.read_csv('files/TestDataSMT.csv')
                    dataAppend = dfSmt.values
                    with open('files/CandidateSetBranch.csv', 'a', newline='') as csvfile:
                        writer = cv.writer(csvfile)
                        writer.writerows(dataAppend)
