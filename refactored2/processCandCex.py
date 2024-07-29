import pandas as pd
import csv as cv
from refactored2 import Pruning
import numpy as np


def funcAddCex2CandidateSet():
    df = pd.read_csv('files/TestDataSMT.csv')
    data = df.values
    with open('files/CandidateSet.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)
        
        
def funcAddCexPruneCandidateSet(tree_model):
    df = pd.read_csv('files/TestDataSMT.csv')
    data = df.values
    
    with open('files/TestDataSMTMain.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)
    
    df = pd.read_csv('files/OracleData.csv')
    #Pruning by negating the data instance
    Pruning.funcPrunInst(df, False)
    dfInst = pd.read_csv('files/CandidateSetInst.csv')
    dataInst = dfInst.values
    with open('files/CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataInst)
    
      
    #Pruning by toggling the branch conditions
    Pruning.funcPrunBranch(df, tree_model)
    dfBranch = pd.read_csv('files/CandidateSetBranch.csv')
    dataBranch = dfBranch.values    
    with open('files/CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataBranch)  


def funcCheckDuplicate(pairfirst, pairsecond, testMatrix):
    pairfirstList = pairfirst.tolist()
    pairsecondList = pairsecond.tolist()
    testDataList = testMatrix.tolist()
    
    for i in range(0, len(testDataList)-1):
        if(pairfirstList == testDataList[i]):
            if(pairsecondList == testDataList[i+1]):
                return True
    
    dfTest = pd.read_csv('files/TestSet.csv')
    dataTest = dfTest.values
    dataTestList = dataTest.tolist()
    for i in range(0, len(dataTestList)-1):
        if(pairfirstList == dataTestList[i]):
            if(pairsecondList == dataTestList[i+1]):
                return True
    return False 


def funcCheckCex():
    dfCandidate = pd.read_csv('files/CandidateSet.csv')
    dataCandidate = dfCandidate.values
    testMatrix = np.zeros((dfCandidate.shape[0], dfCandidate.shape[1]))
    
    candIndx = 0
    testIndx = 0
    
    while(candIndx < dfCandidate.shape[0]-1):
        pairfirst = dataCandidate[candIndx]
        pairsecond = dataCandidate[candIndx+1]
        if(funcCheckDuplicate(pairfirst, pairsecond, testMatrix)):
            candIndx = candIndx+2
        else:
            for k in range(0, dfCandidate.shape[1]):
                testMatrix[testIndx][k] = dataCandidate[candIndx][k]
                testMatrix[testIndx+1][k] = dataCandidate[candIndx+1][k]
            testIndx = testIndx+2    
            candIndx = candIndx+2  

    with open('files/TestSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(testMatrix)
    
    with open('files/Cand-Set.csv', 'w', newline='') as csvfile:
        fieldnames = dfCandidate.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(testMatrix)    

    dfTest = pd.read_csv('files/TestSet.csv')
    dfTest = dfTest[(dfTest.T != 0).any()]
    dfTest.to_csv('files/TestSet.csv', index = False, header = True)  

    dfCand = pd.read_csv('files/Cand-Set.csv')
    dfCand = dfCand[(dfCand.T != 0).any()]
    dfCand.to_csv('files/Cand-Set.csv', index = False, header = True)


def funcAddCexPruneCandidateSet4DNN():
    df = pd.read_csv('files/TestDataSMT.csv')
    data = df.values
    
    with open('files/TestDataSMTMain.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)
    
    df = pd.read_csv('files/OracleData.csv')
    Pruning.funcPrunInst(df, True)
    dfInst = pd.read_csv('files/CandidateSetInst.csv')
    dataInst = dfInst.values
    with open('files/CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataInst)
    
        