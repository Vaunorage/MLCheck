import pandas as pd
import csv as cv
import numpy as np

from refactored import utils
import os
from sklearn.tree import _tree


def getDataType(value, dfOrig, i):
    data_type = str(dfOrig.dtypes[i])
    if ('int' in data_type):
        digit = int(value)
    elif ('float' in data_type):
        digit = float(value)
    return digit


def funcAddCond2File(index, condition_file_path='ConditionFile.txt', dec_smt_file_path='DecSmt.smt2',
                     toggle_branch_file_path='ToggleBranchSmt.smt2'):
    with open(condition_file_path) as fileCond:
        condition_file_content = fileCond.readlines()
    condition_file_content = [x.strip() for x in condition_file_content]

    with open(dec_smt_file_path) as fileSmt:
        smt_file_content = fileSmt.readlines()

    smt_file_content = [x.strip() for x in smt_file_content]
    smt_file_lines = utils.file_len(dec_smt_file_path)
    fileCondSmt = open(toggle_branch_file_path, 'w')

    for i in range(smt_file_lines):
        fileCondSmt.write(smt_file_content[i])
        fileCondSmt.write("\n")
    fileCondSmt.close()

    with open(toggle_branch_file_path, 'r') as fileCondSmt:
        text = fileCondSmt.read()
        text = text.replace("(check-sat)", '')
        text = text.replace("(get-model)", '')

        with open(toggle_branch_file_path, 'w') as fileCondSmt:
            fileCondSmt.write(text)

    fileCondSmt = open(toggle_branch_file_path, 'a')

    temp_cond_content = condition_file_content[index]

    fileCondSmt.write("(assert (not " + temp_cond_content + "))")
    fileCondSmt.write("\n")

    fileCondSmt.write("(check-sat) \n")
    fileCondSmt.write("(get-model) \n")

    fileCondSmt.close()


def funcWrite2File(file_name, toggle_feature_file_path):
    with open(file_name) as fileSmt:
        smt_file_content = fileSmt.readlines()

    smt_file_content = [x.strip() for x in smt_file_content]

    smt_file_lines = utils.file_len(file_name)

    fileTogFeSmt = open(toggle_feature_file_path, 'w')

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

    dfT = pd.read_csv('../TestDataSMTMain.csv')
    pred_arr = np.zeros((tree_.n_outputs))

    node = 0
    depth = 1
    f1 = open('../SampleFile.txt', 'w')
    f1.write("(assert (=> (and ")
    pathCondFile = open('../ConditionFile.txt', 'w')
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

    dfT = pd.read_csv('../TestDataSMTMain.csv')

    node = 0
    depth = 1
    f1 = open('../SampleFile.txt', 'w')
    f1.write("(assert (=> (and ")
    pathCondFile = open('../ConditionFile.txt', 'w')

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


def funcPrunInst(dfOrig, dnn_flag, params,
                 z3_df_output_file_path,
                 final_output_file_path,
                 dnn_smt_file_path='DNNSmt.smt2',
                 dec_smt_file_path='DecSmt.smt2',
                 candidate_set_inst_file_path='CandidateSetInst.csv',
                 test_data_smt_main_file_path='TestDataSMTMain.csv',
                 toggle_feature_file_path='ToggleFeatureSmt.smt2'):
    with open(candidate_set_inst_file_path, 'w', newline='') as csvfile:
        fieldnames = dfOrig.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)

    if (params['multi_label'] == 'True'):
        noClass = int(params['no_of_class'])
    else:
        noClass = 1

    dfRead = pd.read_csv(test_data_smt_main_file_path)
    dataRead = dfRead.values

    for j in range(0, dfRead.shape[0]):
        for i in range(0, dfRead.columns.values.shape[0] - noClass):
            if (dnn_flag == True):
                funcWrite2File(dnn_smt_file_path, toggle_feature_file_path)
            else:
                funcWrite2File(dec_smt_file_path, toggle_feature_file_path)

            with open(toggle_feature_file_path, 'r') as file:
                text = file.read()
                text = text.replace("(check-sat)", '')
                text = text.replace("(get-model)", '')
                with open(toggle_feature_file_path, 'w') as file:
                    file.write(text)

            fileTogFe = open(toggle_feature_file_path, 'a')
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
            if ((int(params['no_of_params']) == 1) and (params['multi_label'] == 'True') and (
                    params['white_box_model'] == 'Decision tree')):
                fileTogFe.write("(assert (not (= " + name + " " + digit + "))) \n")
            else:
                fileTogFe.write("(assert (not (= " + name + str(j) + " " + digit + "))) \n")
            fileTogFe.write("(check-sat) \n")
            fileTogFe.write("(get-model) \n")
            fileTogFe.close()

            os.system(f"z3 {toggle_feature_file_path} > {final_output_file_path}")

            satFlag = utils.funcConvZ3OutToData(dfOrig, params, final_output_file_path,
                                                           z3_df_output_file_path)

            if (satFlag == True):
                dfSmt = pd.read_csv(z3_df_output_file_path)
                dataAppend = dfSmt.values
                with open(candidate_set_inst_file_path, 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(dataAppend)


def funcPrunBranch(dfOrig, tree_model, params,
                   final_output_file_path,
                   tree_output_file_path,
                   z3_df_output_file_path,
                   toggle_feature_file_path,
                   toggle_branch_file_path,
                   candidate_set_branch_file_path,
                   test_data_smt_main_file_path,
                   condition_file_path,
                   dec_smt_file_path):
    with open(candidate_set_branch_file_path, 'w', newline='') as csvfile:
        fieldnames = dfOrig.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)

    dfRead = pd.read_csv(test_data_smt_main_file_path)

    for row in range(0, dfRead.shape[0]):
        if (params['multi_label'] == 'True'):
            funcgetPath4multiLbl(tree_model, dfOrig, row, int(params['no_of_params']))
        else:
            funcgetPath(tree_model, dfOrig, row)
        fileCond = open(tree_output_file_path, 'r')
        first = fileCond.read(1)

        if not first:
            print('No Branch')
        else:
            noPathCond = utils.file_len(condition_file_path)
            if (noPathCond == 'empty'):
                return
            for i in range(noPathCond):
                funcAddCond2File(i, condition_file_path, dec_smt_file_path, toggle_branch_file_path)
                os.system(f"z3 {toggle_feature_file_path} > {final_output_file_path}")
                satFlag = utils.funcConvZ3OutToData(dfOrig, params, final_output_file_path,
                                                               z3_df_output_file_path)

                if (satFlag == True):
                    dfSmt = pd.read_csv(z3_df_output_file_path)
                    dataAppend = dfSmt.values
                    with open(candidate_set_branch_file_path, 'a', newline='') as csvfile:
                        writer = cv.writer(csvfile)
                        writer.writerows(dataAppend)
