import pandas as pd
import csv as cv
import numpy as np
import random as rd
from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
import re
import torch
import os

from paths import HERE
from refactored2 import trainDecTree, tree2Logic, ReadZ3Output, processCandCex, util, assume2logic, assert2logic
from joblib import dump, load
from refactored2 import multiLabelMain
import time

from refactored2.util import local_save, local_load


class generateData:

    def __init__(self, feNameArr, feTypeArr, minValArr, maxValArr):
        self.nameArr = feNameArr
        self.typeArr = feTypeArr
        self.minArr = minValArr
        self.maxArr = maxValArr
        self.paramDict = local_load('param_dict')
        # print('ss')

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
        f = open('files/MUTWeight.txt', 'r')
        weight_content = f.readline()

        for k in range(0, len(self.nameArr)):
            fe_type = self.typeArr[k]
            if 'int' in fe_type:
                if weight_content == 'False':
                    tempData[0][k] = rd.randint(self.minArr[k], self.maxArr[k])
                else:
                    tempData[0][k] = rd.randint(-99999999999, 9999999999999999)
            else:
                if weight_content == 'False':
                    tempData[0][k] = round(rd.uniform(0, self.maxArr[k]), 1)
                else:
                    tempData[0][k] = round(rd.uniform(-99999999999, 9999999999999), 3)

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

        i = 0
        while i <= tst_pm:
            temp = self.funcGenData()
            flg = self.funcCheckUniq(testMatrix, temp)
            if not flg:
                for j in range(0, len(self.nameArr)):
                    testMatrix[i][j] = temp[0][j]
                i = i + 1

        with open('files/TestingData.csv', 'w', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerow(self.nameArr)
            writer.writerows(testMatrix)

        if self.paramDict['train_data_available']:
            dfTrainData = pd.read_csv(self.paramDict['train_data_loc'])
            self.generateTestTrain(dfTrainData, int(self.paramDict['train_ratio']))

    def generateTestTrain(self, dfTrainData, train_ratio):
        tst_pm = round((train_ratio * dfTrainData.shape[0]) / 100)
        data = dfTrainData.values
        testMatrix = np.zeros(((tst_pm + 1), dfTrainData.shape[1]))
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


class makeOracleData:

    def __init__(self, model):
        self.model = model
        self.paramDict = local_load('param_dict')

    def funcGenOracle(self):
        dfTest = pd.read_csv('files/TestingData.csv')
        dataTest = dfTest.values
        predict_list = np.zeros((1, dfTest.shape[0]))
        X = dataTest[:, :-1]

        if 'numpy.ndarray' in str(type(self.model)):
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                dfTest.loc[i, 'Class'] = int(predict_list[0][i])

        else:
            if self.paramDict['model_type'] == 'Pytorch':
                X = torch.tensor(X, dtype=torch.float32)
                predict_class = []
                for i in range(0, X.shape[0]):
                    predict_prob = self.model(X[i].view(-1, X.shape[1]))
                    predict_class.append(int(torch.argmax(predict_prob)))
                for i in range(0, X.shape[0]):
                    dfTest.loc[i, 'Class'] = predict_class[i]
            else:
                predict_class = self.model.predict(X)
                for i in range(0, X.shape[0]):
                    dfTest.loc[i, 'Class'] = int(predict_class[i])
        dfTest.to_csv('files/OracleData.csv', index=False, header=True)


class propCheck:

    def __init__(self, max_samples=None, deadline=None, model=None, no_of_params=None, mul_cex=False,
                 white_box_model=None, no_of_layers=None, layer_size=None, no_of_class=None,
                 no_EPOCHS=None, model_with_weight=False, train_data_available=False, train_data_loc='',
                 multi_label=False, model_type=None, model_path='', no_of_train=None, train_ratio=None):

        self.paramDict = {}

        if multi_label:
            if no_of_class is None:
                raise Exception('Please provide the number of classes the dataset contain')
            else:
                self.paramDict['no_of_class'] = no_of_class
            multiLabelMain.multiLabelPropCheck(no_of_params=no_of_params, max_samples=max_samples, deadline=deadline,
                                               model=model, no_of_class=no_of_class, mul_cex=mul_cex,
                                               white_box_model=white_box_model, no_EPOCHS=no_EPOCHS,
                                               model_path=model_path,
                                               no_of_train=None, train_ratio=None, model_type=model_type)
        else:
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

            if no_EPOCHS is None:
                self.paramDict['no_EPOCHS'] = 20
            else:
                self.paramDict['no_EPOCHS'] = no_EPOCHS

            if (no_of_params is None) or (no_of_params > 3):
                raise Exception("Please provide a value for no_of_params or the value of it is too big")
            else:
                self.no_of_params = no_of_params
            self.paramDict['no_of_params'] = self.no_of_params
            self.paramDict['mul_cex_opt'] = mul_cex
            self.paramDict['multi_label'] = False

            f = open('files/MUTWeight.txt', 'w')
            if not model_with_weight:
                f.write(str(False))
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

            else:
                dfWeight = pd.read_csv('MUTWeight.csv')
                pred_weight = dfWeight.values
                pred_weight = pred_weight[:, :-1]
                self.model = pred_weight
                f.write(str(True))

            f.close()

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
                local_save(self.paramDict, 'param_dict', force_rewrite=True)
            except IOError:
                print("I/O error")

            df = pd.read_csv(HERE.joinpath('refactored2/Datasets/Adult.csv'))
            feNameArr = df.columns.tolist()
            feTypeArr = df.dtypes.apply(str).tolist()
            minValArr = df.min().tolist()
            maxValArr = df.max().tolist()

            local_save(df.dtypes.apply(str).to_dict(), 'feNameType', force_rewrite=True)

            genDataObj = generateData(feNameArr, feTypeArr, minValArr, maxValArr)
            genDataObj.funcGenerateTestData()
            gen_oracle = makeOracleData(self.model)
            gen_oracle.funcGenOracle()


class runChecker:

    def __init__(self):
        self.df = pd.read_csv('files/OracleData.csv')
        f = open('files/MUTWeight.txt', 'r')
        self.MUTcontent = f.readline()
        f.close()
        self.paramDict = local_load('param_dict')

        if self.MUTcontent == 'False':
            self.model_type = self.paramDict['model_type']
            if 'model_path' in self.paramDict:
                model_path = self.paramDict['model_path']
                self.model = load(model_path)
            else:
                self.model = load('Model/MUT.joblib')
        else:
            dfWeight = pd.read_csv('MUTWeight.csv')
            pred_weight = dfWeight.values
            pred_weight = pred_weight[:, :-1]
            self.model = pred_weight
        with open('files/TestSet.csv', 'w', newline='') as csvfile:
            fieldnames = self.df.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
        with open('files/CexSet.csv', 'w', newline='') as csvfile:
            fieldnames = self.df.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)

    def funcCreateOracle(self):
        dfTest = pd.read_csv('files/TestingData.csv')
        data = dfTest.values
        X = data[:, :-1]
        if self.MUTcontent == 'False':
            predict_class = self.model.predict(X)
            for i in range(0, X.shape[0]):
                dfTest.loc[i, 'Class'] = predict_class[i]
            dfTest.to_csv('files/OracleData.csv', index=False, header=True)
        else:
            predict_list = np.zeros((1, dfTest.shape[0]))
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                dfTest.loc[i, 'Class'] = int(predict_list[0][i])
            dfTest.to_csv('files/OracleData.csv', index=False, header=True)

    def chkPairBel(self, tempMatrix, noAttr):
        firstTest = np.zeros((noAttr,))
        secTest = np.zeros((noAttr,))
        dfT = pd.read_csv('files/TestingSet.csv')
        tstMatrix = dfT.values

        for i in range(0, noAttr):
            firstTest[i] = tempMatrix[0][i]

        firstTestList = firstTest.tolist()
        secTestList = secTest.tolist()
        testMatrixList = tstMatrix.tolist()
        for i in range(0, len(testMatrixList) - 1):
            if firstTestList == testMatrixList[i]:
                if secTestList == testMatrixList[i + 1]:
                    return True
        return False

    def chkAttack(self, target_class):
        cexPair = ()
        dfTest = pd.read_csv('files/TestingSet.csv')
        dataTest = dfTest.values
        i = 0
        X = torch.tensor(dataTest, dtype=torch.float32)
        while i < dfTest.shape[0] - 1:
            predict_prob = self.model(X[i].view(-1, X.shape[1]))
            pred_class = int(torch.argmax(predict_prob))
            if pred_class != target_class:
                cexPair = (X[i])
                print('A counter example is found \n')
                print(cexPair)
                return cexPair, True
            i = i + 1
        return cexPair, False

    def checkWithOracle(self):
        assume_dict = []
        f = open('files/assumeStmnt.txt', 'r')
        p = f.readlines()
        dfTr = pd.read_csv('Datasets/mnist_resized.csv')
        noOfAttr = dfTr.shape[1] - 1
        i = 0
        for lines in p:
            if p == '\n':
                pass
            else:
                for col in dfTr.columns.values:
                    if col in lines:
                        num = float(re.search(r'[+-]?([0-9]*[.])[0-9]+', lines).group(0))
                        assume_dict.append(num)
                        i += 1
        f1 = open('files/assertStmnt.txt', 'r')
        p1 = f1.readlines()
        num1 = float(re.search(r'[+-]?([0-9]*[.])[0-9]+', p1[0]).group(0))
        with open('files/TestingSet.csv', 'w', newline='') as csvfile:
            fieldnames = dfTr.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
        dfAg = pd.read_csv('files/TestingSet.csv')
        dfAg.drop('Class', axis=1, inplace=True)
        dfAg.to_csv('files/TestingSet.csv', index=False, header=True)
        inst_count = 0
        i = 0
        while inst_count < 1000:
            tempMatrix = np.zeros((1, noOfAttr))
            for i in range(0, len(assume_dict)):
                tempMatrix[0][i] = assume_dict[i]
            for i in range(len(assume_dict), noOfAttr):
                fe_type = dfTr.dtypes[i]
                fe_type = str(fe_type)
                if 'int' in fe_type:
                    tempMatrix[0][i] = rd.randint(dfTr.iloc[:, i].min(), dfTr.iloc[:, i].max())
                else:
                    tempMatrix[0][i] = rd.uniform(dfTr.iloc[:, i].min(), dfTr.iloc[:, i].max())
            if not self.chkPairBel(tempMatrix, noOfAttr):
                with open('files/TestingSet.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(tempMatrix)
            inst_count = inst_count + 1
        cexPair, flag = self.chkAttack(num1)
        if flag:
            return cexPair, True
        return cexPair, False

    def funcPrediction(self, X, dfCand, testIndx):
        if self.MUTcontent == 'False':
            if self.MUTcontent == 'False':
                return self.model.predict(util.convDataInst(X, dfCand, testIndx, 1))
        else:
            temp_class = np.sign(np.dot(self.model, X[testIndx]))
            if temp_class < 0:
                return 0
            else:
                return temp_class

    def addModelPred(self):
        dfCexSet = pd.read_csv('files/CexSet.csv')
        dataCex = dfCexSet.values
        if self.MUTcontent == 'False':
            predict_class = self.model.predict(dataCex[:, :-1])
            for i in range(0, dfCexSet.shape[0]):
                dfCexSet.loc[i, 'Class'] = predict_class[i]
        else:
            X = dataCex[:, :-1]
            predict_list = np.zeros((1, dfCexSet.shape[0]))
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                if predict_list[0][i] < 0:
                    predict_list[0][i] = 0
                dfCexSet.loc[i, 'Class'] = predict_list[0][i]
        dfCexSet.to_csv('files/CexSet.csv', index=False, header=True)

    def runWithTree(self):
        retrain_flag = False
        MAX_CAND_ZERO = 5
        count_cand_zero = 0
        count = 0
        self.max_samples = int(self.paramDict['max_samples'])
        self.no_of_params = int(self.paramDict['no_of_params'])
        self.mul_cex = self.paramDict['mul_cex_opt']
        self.deadline = int(self.paramDict['deadlines'])
        start_time = time.time()

        while count < self.max_samples:
            print('count is:', count)
            tree = trainDecTree.functrainDecTree()
            tree2Logic.functree2LogicMain(tree, self.no_of_params)
            util.storeAssumeAssert('DecSmt')
            util.addSatOpt('DecSmt')
            os.system(r"z3 files/DecSmt.txt > files/FinalOutput.txt")
            satFlag = ReadZ3Output.funcConvZ3OutToData(self.df)
            if not satFlag:
                if count == 0:
                    print('No CEX is found by the checker at the first trial')
                    return 0
                elif (count != 0) and (self.mul_cex):
                    dfCexSet = pd.read_csv('files/CexSet.csv')
                    if round(dfCexSet.shape[0] / self.no_of_params) == 0:
                        print('No CEX is found')
                        return 0
                    print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
                    self.addModelPred()
                    return round(dfCexSet.shape[0] / self.no_of_params)
                elif (count != 0) and (self.mul_cex):
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
                        if self.mul_cex:
                            dfCexSet = pd.read_csv('files/CexSet.csv')
                            print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
                            if round(dfCexSet.shape[0] / self.no_of_params) > 0:
                                self.addModelPred()
                            return round(dfCexSet.shape[0] / self.no_of_params) + 1
                        else:
                            print('No CEX is found by the checker')
                            return 0
                else:
                    count = count + round(dfCand.shape[0] / self.no_of_params)

                data = dfCand.values
                X = data[:, :-1]
                y = data[:, -1]
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
                        if self.funcPrediction(X, dfCand, testIndx) == y[testIndx]:
                            temp_store.append(X[testIndx])
                            temp_count += 1
                            testIndx += 1
                        else:
                            retrain_flag = True
                            temp_add_oracle.append(X[testIndx])
                            testIndx += 1
                    if temp_count == self.no_of_params:
                        if self.mul_cex:
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
                    self.funcCreateOracle()

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

    def runPropCheck(self):
        self.max_samples = int(self.paramDict['max_samples'])
        self.no_of_params = int(self.paramDict['no_of_params'])
        self.mul_cex = self.paramDict['mul_cex_opt']
        self.deadline = int(self.paramDict['deadlines'])
        white_box = self.paramDict['white_box_model']

        self.runWithTree()


def Assume(*args):
    grammar = Grammar(
        r"""

    expr        = expr1 / expr2 / expr3 /expr4 /expr5 / expr6 /expr7
    expr1       = expr_dist1 logic_op num_log
    expr2       = expr_dist2 logic_op num_log
    expr3       = classVar ws logic_op ws value
    expr4       = classVarArr ws logic_op ws value
    expr5       = classVar ws logic_op ws classVar
    expr6       = classVarArr ws logic_op ws classVarArr
    expr7       = "True"
    expr_dist1  = op_beg?abs?para_open classVar ws arith_op ws classVar para_close op_end?
    expr_dist2  = op_beg?abs?para_open classVarArr ws arith_op ws classVarArr para_close op_end?
    classVar    = variable brack_open number brack_close
    classVarArr = variable brack_open variable brack_close
    para_open   = "("
    para_close  = ")"
    brack_open  = "["
    brack_close = "]"
    variable    = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    logic_op    = ws (geq / leq / eq / neq / and / lt / gt) ws
    op_beg      = number arith_op
    op_end      = arith_op number
    arith_op    = (add/sub/div/mul)
    abs         = "abs"
    add         = "+"
    sub         = "-"
    div         = "/"
    mul         = "*"
    lt          = "<"
    gt          = ">"
    geq         = ">="
    leq         = "<="
    eq          = "="
    neq         = "!="
    and         = "&"
    ws          = ~"\s*"
    value       = ~"\d+"
    num_log     = ~"[+-]?([0-9]*[.])?[0-9]+"
    number      = ~"[+-]?([0-9]*[.])?[0-9]+"
    """
    )

    tree = grammar.parse(args[0])
    assumeVisitObj = assume2logic.AssumptionVisitor()
    if len(args) == 3:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.storeArr(args[2])
        assumeVisitObj.visit(tree)
    elif len(args) == 2:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.visit(tree)
    elif len(args) == 1:
        assumeVisitObj.visit(tree)


def Assert(*args):
    grammar = Grammar(
        r"""
        expr        = expr1 / expr2/ expr3
        expr1       = classVar ws operator ws number
        expr2       = classVar ws operator ws classVar
        expr3       = classVar mul_cl_var ws operator ws neg? classVar mul_cl_var
        classVar    = class_pred brack_open variable brack_close
        model_name  = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
        class_pred  = model_name classSymbol
        classSymbol = ~".predict"
        brack_open  = "("
        brack_close = ")"
        variable    = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
        brack3open  = "["
        brack3close = "]"
        class_name  = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
        mul_cl_var  = brack3open class_name brack3close
        operator    = ws (gt/ lt/ geq / leq / eq / neq / and/ implies) ws
        lt          = "<"
        gt          = ">"
        geq         = ">="
        implies     = "=>"
        neg         = "~"
        leq         = "<="
        eq          = "=="
        neq         = "!="
        and         = "&"
        ws          = ~"\s*"
        number      = ~"[+-]?([0-9]*[.])?[0-9]+"
        """
    )

    tree = grammar.parse(args[0])
    assertVisitObj = assert2logic.AssertionVisitor()
    assertVisitObj.visit(tree)
    paramDict = local_load('param_dict')
    if paramDict['multi_label']:
        start_time = time.time()
        obj_multi = multiLabelMain.runChecker()
        obj_multi.runPropCheck()
        print('time required is', time.time() - start_time)
    else:
        obj_faircheck = runChecker()
        start_time = time.time()
        obj_faircheck.runPropCheck()
        print('time required is', time.time() - start_time)

    if os.path.exists('files/assumeStmnt.txt'):
        os.remove('files/assumeStmnt.txt')
    if os.path.exists('files/assertStmnt.txt'):
        os.remove('files/assertStmnt.txt')

    if os.path.exists('files/Cand-Set.csv'):
        os.remove('files/Cand-Set.csv')
    if os.path.exists('files/CandidateSet.csv'):
        os.remove('files/CandidateSet.csv')
    if os.path.exists('files/CandidateSetInst.csv'):
        os.remove('files/CandidateSetInst.csv')
    if os.path.exists('files/CandidateSetBranch.csv'):
        os.remove('files/CandidateSetBranch.csv')

    if os.path.exists('files/TestDataSMT.csv'):
        os.remove('files/TestDataSMT.csv')
    if os.path.exists('files/TestDataSMTMain.csv'):
        os.remove('files/TestDataSMTMain.csv')

    if os.path.exists('files/DecSmt.smt2'):
        os.remove('files/DecSmt.smt2')
    if os.path.exists('files/ToggleBranchSmt.smt2'):
        os.remove('files/ToggleBranchSmt.smt2')
    if os.path.exists('files/ToggleFeatureSmt.smt2'):
        os.remove('files/ToggleFeatureSmt.smt2')
    if os.path.exists('files/TreeOutput.txt'):
        os.remove('files/TreeOutput.txt')

    if os.path.exists('files/SampleFile.txt'):
        os.remove('files/SampleFile.txt')
    if os.path.exists('files/FinalOutput.txt'):
        os.remove('files/FinalOutput.txt')
    if os.path.exists('files/MUTWeight.txt'):
        os.remove('files/MUTWeight.txt')
    if os.path.exists('files/ConditionFile.txt'):
        os.remove('files/ConditionFile.txt')

    if os.path.exists('MUTWeight.csv'):
        os.remove('MUTWeight.csv')
    if os.path.exists('files/MUTWeight.txt'):
        os.remove('files/MUTWeight.txt')
    if os.path.exists('DNNSmt.smt2'):
        os.remove('DNNSmt.smt2')

    if os.path.exists('TestData.csv'):
        os.remove('TestData.csv')
    if os.path.exists('TestDataSet.csv'):
        os.remove('TestDataSet.csv')
    if os.path.exists('CandTestDataSet.csv'):
        os.remove('CandTestDataSet.csv')
