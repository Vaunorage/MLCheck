#!/usr/bin/env python


import pandas as pd
import csv as cv
import numpy as np
import random as rd
import sys
from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
import re
import os, time

from paths import HERE
from refactored2 import trainDecTree, tree2Logic, ReadZ3Output
from refactored2 import processCandCex, util
from joblib import dump, load


class generateData:

    def __init__(self, feNameArr, feTypeArr, minValArr, maxValArr):
        self.nameArr = feNameArr
        self.typeArr = feTypeArr
        self.minArr = minValArr
        self.maxArr = maxValArr
        with open('files/param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)

    # function to search for duplicate test data
    def binSearch(self, alist, item):
        if len(alist) == 0:
            return False
        else:
            midpoint = len(alist) // 2
            if alist[midpoint] == item:
                return True
            else:
                if item < alist[midpoint]:
                    return self.binSearch(alist[:midpoint], item)
                else:
                    return self.binSearch(alist[midpoint + 1:], item)

    # Function to generate a new sample
    def funcGenData(self):
        tempData = np.zeros((1, len(self.nameArr)), dtype=object)
        for k in range(0, len(self.nameArr)):
            fe_type = ''
            fe_type = self.typeArr[k]
            if 'int' in fe_type:
                tempData[0][k] = rd.randint(self.minArr[k], self.maxArr[k])
            else:
                tempData[0][k] = round(rd.uniform(0, self.maxArr[k]), 1)

        return tempData

    # Function to check whether a newly generated sample already exists in the list of samples
    def funcCheckUniq(self, matrix, row):
        row_temp = row.tolist()
        matrix_new = matrix.tolist()
        if row_temp in matrix_new:
            return True
        else:
            return False

    # Function to combine several steps
    def funcGenerateTestData(self):
        tst_pm = int(self.paramDict['no_of_train'])
        testMatrix = np.zeros(((tst_pm + 1), len(self.nameArr)), dtype=object)
        feature_track = []
        flg = False

        i = 0
        while i <= tst_pm:
            # Generating a test sample
            temp = self.funcGenData()
            # Checking whether that sample already in the test dataset
            flg = self.funcCheckUniq(testMatrix, temp)
            if not flg:
                for j in range(0, len(self.nameArr)):
                    testMatrix[i][j] = temp[0][j]
                i = i + 1

        with open('files/TestingData.csv', 'w', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerow(self.nameArr)
            writer.writerows(testMatrix)

        if self.paramDict['train_data_available'] == 'True':
            dfTrainData = pd.read_csv(self.paramDict['train_data_loc'])
            self.generateTestTrain(dfTrainData, int(self.paramDict['train_ratio']))

    # Function to take train data as test data
    def generateTestTrain(self, dfTrainData, train_ratio):
        tst_pm = round((train_ratio * dfTrainData.shape[0]) / 100)
        data = dfTrainData.values
        testMatrix = np.zeros(((tst_pm + 1), dfTrainData.shape[1]))
        flg = True
        testCount = 0
        ratioTrack = []
        noOfRows = dfTrainData.shape[0]
        while testCount <= tst_pm:
            ratio = rd.randint(0, noOfRows - 1)
            if testCount >= 1:
                flg = self.binSearch(ratioTrack, ratio)
                if not flg:
                    ratioTrack.append(ratio)
                    testMatrix[testCount] = data[ratio]
                    testCount = testCount + 1
            if testCount == 0:
                ratioTrack.append(ratio)
                testMatrix[testCount] = data[ratio]
                testCount = testCount + 1
        with open('files/TestingData.csv', 'a', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerows(testMatrix)


class dataFrameCreate(NodeVisitor):

    def __init__(self):
        self.feName = None
        self.feType = None
        self.feMinVal = -99999
        self.feMaxVal = 0

    def generic_visit(self, node, children):
        pass

    def visit_feName(self, node, children):
        self.feName = node.text

    def visit_feType(self, node, children):
        self.feType = node.text

    def visit_minimum(self, node, children):
        digit = float(re.search(r'\d+', node.text).group(0))
        self.feMinVal = digit

    def visit_maximum(self, node, children):
        digit = float(re.search(r'\d+', node.text).group(0))
        self.feMaxVal = digit


class readXmlFile:
    def __init__(self, fileName):
        self.fileName = fileName

    def funcReadXml(self):
        grammar = Grammar(
            r"""
        
            expr             = name / type / minimum / maximum / xmlStartDoc / xmlStartInps / xmlEndInps / xmlStartInp /
                                                                        xmlEndInp / xmlStartValTag /xmlEndValTag
            name             = xmlStartNameTag feName xmlEndNameTag
            type             = xmlStartTypeTag feType xmlEndTypeTag
            minimum          = xmlStartMinTag number xmlEndMinTag
            maximum          = xmlStartMaxTag number xmlEndMaxTag
            xmlStartDoc      = '<?xml version="1.0" encoding="UTF-8"?>'
            xmlStartInps     = "<Inputs>"
            xmlEndInps       = "<\Inputs>"
            xmlStartInp      = "<Input>"
            xmlEndInp        = "<\Input>"
            xmlStartNameTag  = "<Feature-name>"
            xmlEndNameTag    = "<\Feature-name>"
            xmlStartTypeTag  = "<Feature-type>"
            xmlEndTypeTag    = "<\Feature-type>"
            xmlStartValTag   = "<Value>"
            xmlEndValTag     = "<\Value>"
            xmlStartMinTag   = "<minVal>"
            xmlEndMinTag     = "<\minVal>"
            xmlStartMaxTag   = "<maxVal>"
            xmlEndMaxTag     = "<\maxVal>"
            feName           = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
            feType           = ~"[A-Z 0-9]*"i
            number           = ~"[+-]?([0-9]*[.])?[0-9]+"
            """
        )

        with open(self.fileName) as f1:
            file_content = f1.readlines()
        file_content = [x.strip() for x in file_content]

        feNameArr = []
        feTypeArr = []
        minValArr = []
        maxValArr = []
        for lines in file_content:
            tree = grammar.parse(lines)
            dfObj = dataFrameCreate()
            dfObj.visit(tree)
            if dfObj.feName is not None:
                feNameArr.append(dfObj.feName)
            if dfObj.feType is not None:
                feTypeArr.append(dfObj.feType)
            if dfObj.feMinVal != -99999:
                minValArr.append(dfObj.feMinVal)
            if dfObj.feMaxVal != 0:
                maxValArr.append(dfObj.feMaxVal)

        genDataObj = generateData(feNameArr, feTypeArr, minValArr, maxValArr)
        genDataObj.funcGenerateTestData()


class makeOracleData:
    def __init__(self, model, train_data, train_data_loc):
        self.model = model
        with open('files/param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)

    def funcGenOracle(self):
        noOfClasses = int(self.paramDict['no_of_class'])
        dfTest = pd.read_csv('files/TestingData.csv')
        dataTest = dfTest.values
        X = dataTest[:, :-noOfClasses]
        predict_class = self.model.predict(X)
        with open('PredictClass.csv', 'w', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerows(predict_class)

        for i in range(0, noOfClasses):
            className = str(dfTest.columns.values[dfTest.shape[1] - noOfClasses + i])
            for j in range(0, X.shape[0]):
                dfTest.loc[j, className] = predict_class[j][i]
        dfTest.to_csv('files/OracleData.csv', index=False, header=True)


class multiLabelPropCheck:

    def __init__(self, max_samples=None, deadline=None, model=None, no_of_params=None,
                 mul_cex=False, white_box_model=None, no_of_class=None,
                 no_EPOCHS=None, train_data_available=False, train_data_loc='', model_path='', no_of_train=None,
                 train_ratio=None, model_type=None):

        self.paramDict = {}
        if max_samples is None:
            self.max_samples = 1000
        else:
            self.max_samples = max_samples
        self.paramDict['max_samples'] = self.max_samples

        if deadline is None:
            self.deadline = 500000
        else:
            self.deadline = deadline
        self.paramDict['deadlines'] = self.deadline

        if white_box_model is None:
            self.white_box_model = 'Decision tree'
        else:
            self.white_box_model = white_box_model
        self.paramDict['white_box_model'] = self.white_box_model
        self.paramDict['no_of_class'] = no_of_class

        if no_EPOCHS is None:
            self.paramDict['no_EPOCHS'] = 100
        else:
            self.paramDict['no_EPOCHS'] = no_EPOCHS

        if (no_of_params is None) or (no_of_params > 3):
            raise Exception("Please provide a value for no_of_params or the value of it is too big")
        else:
            self.no_of_params = no_of_params
        self.paramDict['no_of_params'] = self.no_of_params
        self.paramDict['mul_cex_opt'] = mul_cex
        self.paramDict['multi_label'] = True

        if model_type == 'sklearn':
            if model is None:
                if model_path == '':
                    raise Exception("Please provide a classifier to check")
                else:
                    self.model = load(model_path)
                    self.paramDict['model_path'] = model_path
                    self.paramDict['model_type'] = 'sklearn'

            else:
                self.paramDict['model_type'] = 'sklearn'
                self.model = model
                dump(self.model, 'Model/MUT.joblib')

        else:
            raise Exception("Please provide the type of the model (Pytorch/sklearn)")

        if no_of_train is None:
            self.no_of_train = 1000
        else:
            self.no_of_train = no_of_train
        if train_data_available:
            if train_data_loc == '':
                raise Exception('Please provide the training data location')
            else:
                if train_ratio is None:
                    self.paramDict['train_ratio'] = 100
                else:
                    self.paramDict['train_ratio'] = train_ratio
        self.paramDict['no_of_train'] = self.no_of_train
        self.paramDict['train_data_available'] = train_data_available
        self.paramDict['train_data_loc'] = train_data_loc

        try:
            with open('files/param_dict.csv', 'w') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in self.paramDict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")

        df = pd.read_csv(HERE.joinpath('refactored2/Datasets/Adult.csv'))
        feNameArr = df.columns.tolist()
        feTypeArr = df.dtypes.apply(str).tolist()
        minValArr = df.min().tolist()
        maxValArr = df.max().tolist()
        genDataObj = generateData(feNameArr, feTypeArr, minValArr, maxValArr)
        genDataObj.funcGenerateTestData()

        genOrcl = makeOracleData(self.model, train_data_available, train_data_loc)
        genOrcl.funcGenOracle()


class runChecker:
    def __init__(self):
        self.df = pd.read_csv('files/OracleData.csv')
        with open('files/param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)
        if 'model_path' in self.paramDict:
            self.model = load(self.paramDict['model_path'])
        else:
            self.model = load('Model/MUT.joblib')
        with open('files/TestSet.csv', 'w', newline='') as csvfile:
            fieldnames = self.df.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
        with open('files/CexSet.csv', 'w', newline='') as csvfile:
            fieldnames = self.df.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
        self.max_samples = int(self.paramDict['max_samples'])
        self.no_of_params = int(self.paramDict['no_of_params'])
        self.mul_cex = self.paramDict['mul_cex_opt']
        self.deadline = int(self.paramDict['deadlines'])
        self.white_box = self.paramDict['white_box_model']
        self.no_of_class = int(self.paramDict['no_of_class'])

    def get_index(self, name):
        df = pd.read_csv('files/OracleData.csv')
        for i in range(0, df.shape[1] - 1):
            if name == df.columns.values[i]:
                return i

    def checkWithOracle(self):
        df = pd.read_csv('files/OracleData.csv')
        data_oracle = self.df.values
        data_oracle = data_oracle[:, :-self.no_of_class]
        classNameList = []
        indices = []
        flag_act_per = False
        flag_act_obj = False
        flag_per_obj = False
        index = self.df.shape[1] - self.no_of_class
        for i in range(0, self.no_of_class):
            classNameList.append(str(self.df.columns.values[index + i]))
        with open('files/assertStmnt.txt') as f1:
            file_content = f1.readlines()
        statement = [x.strip() for x in file_content]

        for name in classNameList:
            for st in statement:
                if name in st:
                    indices.append(self.get_index(name))

        pred_arr = self.model.predict(data_oracle)
        for i in range(0, 2):
            if (indices[0] == 50) & (indices[1] == 51):
                flag_act_per = True
            elif (indices[0] == 51) & (indices[1] == 52):
                flag_act_obj = True
            elif (indices[0] == 50) & (indices[1] == 52):
                flag_per_obj = True
        if flag_act_per:
            for i in range(0, len(pred_arr)):
                if ((pred_arr[i][1] == 1) & (pred_arr[i][0] == 0)):
                    print('A CEX is found')
                    print(data_oracle[i])
                    with open('files/CexSet.csv', 'a', newline='') as csvfile:
                        writer = cv.writer(csvfile)
                        writer.writerows(np.reshape(data_oracle[i], (1, df.shape[1] - self.no_of_class)))
                    return True
                else:
                    return False
        elif flag_per_obj:
            for i in range(0, len(pred_arr)):
                if ((pred_arr[i][0] == 1) & (pred_arr[i][2] == 1)):
                    print('A CEX is found')
                    print(data_oracle[i])
                    with open('files/CexSet.csv', 'a', newline='') as csvfile:
                        writer = cv.writer(csvfile)
                        writer.writerows(np.reshape(data_oracle[i], (1, df.shape[1] - self.no_of_class)))
                    return True
                else:
                    return False
        elif flag_act_obj:
            for i in range(0, len(pred_arr)):
                if ((pred_arr[i][1] == 1) & (pred_arr[i][2] == 1)):
                    print('A CEX is found')
                    print(data_oracle[i])
                    with open('files/CexSet.csv', 'a', newline='') as csvfile:
                        writer = cv.writer(csvfile)
                        writer.writerows(np.reshape(data_oracle[i], (1, df.shape[1] - self.no_of_class)))
                    return True
                else:
                    return False
        return False

    def funcPrediction(self, X, dfCand, testIndx, y):
        pred_arr = self.model.predict(util.convDataInst(X, dfCand, testIndx, self.no_of_class))
        for i in range(0, self.no_of_class):
            if pred_arr[0][i] == y[testIndx][i]:
                pass
            else:
                return False
        return True

    def addModelPred(self):
        dfCexSet = pd.read_csv('files/CexSet.csv')
        dataCex = dfCexSet.values
        X = dataCex[:, :-self.no_of_class]
        predict_class = self.model.predict(X)
        index = self.df.shape[1] - self.no_of_class
        for i in range(0, self.no_of_class):
            className = str(self.df.columns.values[index + i])
            for j in range(0, X.shape[0]):
                dfCexSet.loc[j, className] = predict_class[j][i]
        dfCexSet.to_csv('files/CexSet.csv', index=False, header=True)

    def runPropCheck(self):
        retrain_flag = False
        MAX_CAND_ZERO = 5
        count_cand_zero = 0
        count = 0
        satFlag = False
        start_time = time.time()

        while count < self.max_samples:
            count = count + 1
            print('count in multi:', count)
            tree = trainDecTree.functrainDecTree(self.no_of_class)

            tree2Logic.functree2LogicMain(tree, self.no_of_params)
            util.storeAssumeAssert('files/DecSmt.smt2')
            util.addSatOpt('files/DecSmt.smt2')
            os.system(r"z3 files/DecSmt.smt2 > files/FinalOutput.txt")
            satFlag = ReadZ3Output.funcConvZ3OutToData(self.df)
            if not satFlag:
                if count == 0:
                    print('No CEX is found by the checker at the first trial')
                    return 0
                elif (count != 0) and (self.mul_cex == 'True'):
                    dfCexSet = pd.read_csv('files/CexSet.csv')
                    if round(dfCexSet.shape[0] / self.no_of_params) == 0:
                        print('No CEX is found')
                        return 0
                    print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
                    self.addModelPred()
                    return round(dfCexSet.shape[0] / self.no_of_params)
                elif (count != 0) and (self.mul_cex == 'False'):
                    print('No Cex is found after ' + str(count) + ' no. of trials')
                    return 0

            else:
                processCandCex.funcAddCex2CandidateSet()
                processCandCex.funcAddCexPruneCandidateSet(tree)
                processCandCex.funcCheckCex()
                # Increase the count if no further candidate cex has been found
                dfCand = pd.read_csv('files/Cand-Set.csv')
                if round(dfCand.shape[0] / self.no_of_params) == 0:
                    count_cand_zero += 1
                    if count_cand_zero == MAX_CAND_ZERO:
                        if self.mul_cex == 'True':
                            dfCexSet = pd.read_csv('files/CexSet.csv')
                            print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
                            if round(dfCexSet.shape[0] / self.no_of_params) > 0:
                                self.addModelPred()
                            return round(dfCexSet.shape[0] / self.no_of_params)
                        else:
                            print('No CEX is found by the checker')
                            return 0
                else:
                    count = count + round(dfCand.shape[0] / self.no_of_params)

                data = dfCand.values
                X = data[:, :-self.no_of_class]
                y = data[:, -self.no_of_class:]
                if dfCand.shape[0] % self.no_of_params == 0:
                    arr_length = dfCand.shape[0]
                else:
                    arr_length = dfCand.shape[0] - 1
                testIndx = 0

                while testIndx < arr_length:
                    temp_count = 0
                    temp_store = []
                    temp_add_oracle = []
                    for i in range(0, self.no_of_params):
                        if self.funcPrediction(X, dfCand, testIndx, y):
                            temp_store.append(X[testIndx])
                            temp_count += 1
                            testIndx += 1
                        else:
                            retrain_flag = True
                            temp_add_oracle.append(X[testIndx])
                            testIndx += 1
                    if temp_count == self.no_of_params:
                        if self.mul_cex == 'True':
                            with open('files/CexSet.csv', 'a', newline='') as csvfile:
                                writer = cv.writer(csvfile)
                                writer.writerows(temp_store)
                        else:
                            print('A counter example is found, check it in files/CexSet.csv file: ', temp_store)
                            with open('files/CexSet.csv', 'a', newline='') as csvfile:
                                writer = cv.writer(csvfile)
                                writer.writerows(temp_store)
                            self.addModelPred()
                            return 1
                    else:
                        util.funcAdd2Oracle(temp_add_oracle)

                if retrain_flag:
                    util.funcCreateOracle(self.no_of_class, True, self.model)
                if (time.time() - start_time) > self.deadline:
                    print("Time out")
                    break

            dfCexSet = pd.read_csv('files/CexSet.csv')
            if (round(dfCexSet.shape[0] / self.no_of_params) > 0) and (count >= self.max_samples):
                self.addModelPred()
                print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
                print('No. of Samples looked for counter example has exceeded the max_samples limit')
            else:
                print('No counter example has been found')
