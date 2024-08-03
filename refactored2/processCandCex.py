import pandas as pd
from refactored2 import Pruning
import numpy as np

from refactored2.util import local_load, local_save


def funcAddCex2CandidateSet():
    df = local_load('TestDataSMT')
    local_save(df, 'CandidateSet', force_rewrite=True)

def funcAddCexPruneCandidateSet(tree_model):
    df = local_load('TestDataSMT')

    local_save(df, 'TestDataSMTMain', force_rewrite=True)

    df = local_load('OracleData')
    # Pruning by negating the data instance
    Pruning.funcPrunInst(df, False)
    dfInst = local_load('CandidateSetInst')
    local_save(dfInst, 'CandidateSet')

    # Pruning by toggling the branch conditions
    Pruning.funcPrunBranch(df, tree_model)
    dfBranch = local_load('CandidateSetBranch')
    local_save(dfBranch, 'CandidateSet')


def funcCheckDuplicate(pairfirst, pairsecond, testMatrix):
    pairfirstList = pairfirst.tolist()
    pairsecondList = pairsecond.tolist()
    testDataList = testMatrix.tolist()

    for i in range(0, len(testDataList) - 1):
        if (pairfirstList == testDataList[i]):
            if (pairsecondList == testDataList[i + 1]):
                return True

    dfTest = pd.read_csv('files/TestSet.csv')
    dataTest = dfTest.values
    dataTestList = dataTest.tolist()
    for i in range(0, len(dataTestList) - 1):
        if (pairfirstList == dataTestList[i]):
            if (pairsecondList == dataTestList[i + 1]):
                return True
    return False


def funcCheckCex():
    dfCandidate = local_load('CandidateSet')
    dataCandidate = dfCandidate.values
    testMatrix = np.zeros((dfCandidate.shape[0], dfCandidate.shape[1]))

    candIndx = 0
    testIndx = 0

    while (candIndx < dfCandidate.shape[0] - 1):
        pairfirst = dataCandidate[candIndx]
        pairsecond = dataCandidate[candIndx + 1]
        if (funcCheckDuplicate(pairfirst, pairsecond, testMatrix)):
            candIndx = candIndx + 2
        else:
            for k in range(0, dfCandidate.shape[1]):
                testMatrix[testIndx][k] = dataCandidate[candIndx][k]
                testMatrix[testIndx + 1][k] = dataCandidate[candIndx + 1][k]
            testIndx = testIndx + 2
            candIndx = candIndx + 2

    local_save(pd.DataFrame(testMatrix), 'TestSet')

    local_save(pd.DataFrame(testMatrix, columns=dfCandidate.columns.values), 'Cand-Set')

    dfTest = local_load('TestSet')
    dfTest = dfTest[(dfTest.T != 0).any()]
    local_save(dfTest, 'TestSet', force_rewrite=True)

    dfCand = local_load('Cand-Set')
    dfCand = dfCand[(dfCand.T != 0).any()]
    local_save(dfCand, 'Cand-Set', force_rewrite=True)
