#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import csv as cv
from refactored import prunning
import numpy as np


# In[1]:


def funcAddCex2CandidateSet(z3_output_file_path, candidates_file_path):
    df = pd.read_csv(z3_output_file_path)
    data = df.values
    with open(candidates_file_path, 'w', newline='') as csvfile:
        fieldnames = df.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)


def funcAddCexPruneCandidateSet(tree_model, params, z3_df_output_file_path, test_smt_file_path,
                                tree_output_file_path, candidate_set_inst_file_path, candidate_set_branch_file_path,
                                candidate_set_file_path, final_output_file_path, dnn_smt_file_path, dec_smt_file_path,
                                toggle_feature_file_path, toggle_branch_smt_file_path, condition_file_path):
    df = pd.read_csv(z3_df_output_file_path)
    data = df.values

    with open(test_smt_file_path, 'w', newline='') as csvfile:
        fieldnames = df.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)

    prunning.funcPrunInst(df, False, params, z3_df_output_file_path=z3_df_output_file_path,
                                     final_output_file_path=final_output_file_path,
                                     dnn_smt_file_path=dnn_smt_file_path,
                                     dec_smt_file_path=dec_smt_file_path,
                                     test_data_smt_main_file_path=test_smt_file_path,
                                     toggle_feature_file_path=toggle_feature_file_path,
                                     candidate_set_inst_file_path=candidate_set_inst_file_path
                                     )
    dfInst = pd.read_csv(candidate_set_inst_file_path)
    dataInst = dfInst.values
    with open(candidate_set_file_path, 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataInst)

    prunning.funcPrunBranch(df, tree_model, params, final_output_file_path,
                                       tree_output_file_path,
                                       z3_df_output_file_path,
                                       toggle_feature_file_path=toggle_feature_file_path,
                                       toggle_branch_file_path=toggle_branch_smt_file_path,
                                       candidate_set_branch_file_path=candidate_set_branch_file_path,
                                       test_data_smt_main_file_path=test_smt_file_path,
                                       condition_file_path=condition_file_path,
                                       dec_smt_file_path=dec_smt_file_path)
    dfBranch = pd.read_csv(candidate_set_branch_file_path)
    dataBranch = dfBranch.values
    with open(candidate_set_file_path, 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataBranch)


def funcCheckDuplicate(pairfirst, pairsecond, testMatrix, test_set_file_path='TestSet.csv'):
    pairfirstList = pairfirst.tolist()
    pairsecondList = pairsecond.tolist()
    testDataList = testMatrix.tolist()

    for i in range(0, len(testDataList) - 1):
        if (pairfirstList == testDataList[i]):
            if (pairsecondList == testDataList[i + 1]):
                return True

    dfTest = pd.read_csv(test_set_file_path)
    dataTest = dfTest.values
    dataTestList = dataTest.tolist()
    for i in range(0, len(dataTestList) - 1):
        if (pairfirstList == dataTestList[i]):
            if (pairsecondList == dataTestList[i + 1]):
                return True
    return False


def funcCheckCex(candidates_file_path, test_set_file_path, cand_set_file_path):
    dfCandidate = pd.read_csv(candidates_file_path)
    dataCandidate = dfCandidate.values
    testMatrix = np.zeros((dfCandidate.shape[0], dfCandidate.shape[1]))

    candIndx = 0
    testIndx = 0

    while (candIndx < dfCandidate.shape[0] - 1):
        pairfirst = dataCandidate[candIndx]
        pairsecond = dataCandidate[candIndx + 1]

        if (funcCheckDuplicate(pairfirst, pairsecond, testMatrix, test_set_file_path)):
            candIndx = candIndx + 2
        else:
            for k in range(0, dfCandidate.shape[1]):
                testMatrix[testIndx][k] = dataCandidate[candIndx][k]
                testMatrix[testIndx + 1][k] = dataCandidate[candIndx + 1][k]
            testIndx = testIndx + 2
            candIndx = candIndx + 2

    with open(test_set_file_path, 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(testMatrix)

    with open(cand_set_file_path, 'w', newline='') as csvfile:
        fieldnames = dfCandidate.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(testMatrix)

        # Eliminating the rows with zero values
    dfTest = pd.read_csv(test_set_file_path)
    dfTest = dfTest[(dfTest.T != 0).any()]
    dfTest.to_csv(test_set_file_path, index=False, header=True)

    # Eliminating the rows with zero values
    dfCand = pd.read_csv(cand_set_file_path)
    dfCand = dfCand[(dfCand.T != 0).any()]
    dfCand.to_csv(cand_set_file_path, index=False, header=True)


def funcAddCexPruneCandidateSet4DNN():
    df = pd.read_csv('../TestDataSMT.csv')
    data = df.values

    with open('../TestDataSMTMain.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)

    df = pd.read_csv('../OracleData.csv')
    prunning.funcPrunInst(df, True, )
    dfInst = pd.read_csv('../CandidateSetInst.csv')
    dataInst = dfInst.values
    with open('../CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataInst)
