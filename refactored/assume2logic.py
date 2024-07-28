import time

import pandas as pd
import numpy as np
import random as rd
import csv

from distributed.protocol import torch
from joblib import dump, load
from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
import re
import os
from sklearn.tree import DecisionTreeClassifier

from PytorchDNNStruct import NetArch1
from refactored.dnn import ConvertDNN2logic, functrainDNN
from refactored.processCandCex import funcAddCex2CandidateSet, funcAddCexPruneCandidateSet4DNN, funcCheckCex, \
    funcAddCexPruneCandidateSet
from refactored.tree2logic import functree2LogicMain
from refactored import utils


class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[self.size() - 1]

    def size(self):
        return len(self.items)


class InfixConverter:
    def __init__(self):
        self.stack = Stack()
        self.precedence = {'+': 1, '-': 1, 'abs': 1, '*': 2, '/': 2, '^': 3}

    def hasLessOrEqualPriority(self, a, b):
        if a not in self.precedence:
            return False
        if b not in self.precedence:
            return False
        return self.precedence[a] <= self.precedence[b]

    def isOperand(self, ch):
        return ch.isalpha() or ch.isdigit() or ch in '.' or ch in '_'

    def isOperator(self, x):
        ops = ['+', '-', 'abs', '/', '*']
        return x in ops

    def isOpenParenthesis(self, ch):
        return ch == '('

    def isCloseParenthesis(self, ch):
        return ch == ')'

    def toPostfix(self, expr):
        expr = expr.replace(" ", "")
        self.stack = Stack()
        output = ' '

        for c in expr:
            if self.isOperand(c):
                output += c
            else:
                if self.isOpenParenthesis(c):
                    output += " "
                    output += ')'

                    self.stack.push(c)
                elif self.isCloseParenthesis(c):
                    operator = self.stack.pop()
                    output += " "
                    output += '('

                    while not self.isOpenParenthesis(operator):
                        output += " "
                        output += operator
                        operator = self.stack.pop()
                else:
                    while (not self.stack.isEmpty()) and self.hasLessOrEqualPriority(c, self.stack.peek()):
                        output += " "
                        output += self.stack.pop()

                    self.stack.push(c)
                output += " "
        while (not self.stack.isEmpty()):
            output += self.stack.pop()
        return output

    def toPrefix(self, expr):
        reverse_expr = ''
        for c in expr[::-1]:
            if c == '(':
                reverse_expr += ") "
            elif c == ')':
                reverse_expr += "( "
            else:
                reverse_expr += c + " "

        reverse_postfix = self.toPostfix(reverse_expr)
        return reverse_postfix[::-1]

    def convert(self, expr):
        try:
            result = eval(expr)
        except:
            result = expr
        return self.toPrefix(expr)


class AssumptionVisitor(NodeVisitor):

    def __init__(self, oracle_data, filepath):
        self.varList = []
        self.classVarList = []
        self.arithOperator = []
        self.logicOperator = ""
        self.numList = []
        self.numEnd = ""
        self.feIndex = 99999
        self.feValue = 0
        self.count = 0
        self.df = oracle_data
        self.feArr = []
        self.noOfAttr = self.df.shape[1]
        self.varMapDict = {}
        self.prefix_list = []
        self.varInd = False
        self.arrFlag = False
        self.filepath = filepath
        self.final_var_mapping = []

    def save_assumptions(self, assumption):
        with open(self.filepath, 'a') as file:
            file.write(assumption)

    def generic_visit(self, node, children):
        pass

    def visit_arith_op(self, node, children):
        self.arithOperator.append(node.text)

    def visit_logic_op(self, node, children):
        if ('!=' in node.text):
            self.logicOperator = 'not(='
        else:
            self.logicOperator = node.text

    def visit_number(self, node, children):
        self.numList.append(node.text)

    def visit_classVar(self, node, children):
        if (self.varInd == True):
            raise Exception("Feature indexes given twice")
            sys.exit(1)
        self.classVarList.append(node.text)
        self.checkIndexConstncy()
        self.feIndex = int(re.search(r'\d+', node.text).group(0))
        self.checkValFeIndex()

    def visit_classVarArr(self, node, children):
        self.classVarList.append(node.text)

    def visit_num_log(self, node, children):
        self.numEnd = node.text

    def visit_value(self, node, children):
        self.feValue = float(node.text)

    def visit_expr_dist1(self, node, children):
        expr_dist1 = str(node.text)
        self.getPrefixExp(expr_dist1)

    def visit_expr1(self, node, children):
        self.expr2logic(self.prefix_list)

    def visit_expr_dist2(self, node, children):
        expr_dist2 = str(node.text)
        self.getPrefixExp(expr_dist2)

    def visit_expr2(self, node, children):
        self.expr2logic(self.prefix_list)

    def visit_expr3(self, node, children):
        temp_expr = node.text
        self.replaceIndex(temp_expr)
        assumption = '\n'
        assumption += "(assert (" + self.logicOperator + " " + str(
            self.df.columns.values[self.feIndex] + str(0)) + " " + str(self.feValue) + "))"

        if (self.logicOperator == 'not(='):
            assumption += ')'
        assumption += '\n'
        self.save_assumptions(assumption)

    def visit_expr4(self, node, children):
        temp_expr = node.text
        self.replaceIndex(temp_expr)
        assumption = '\n'
        assumption += "(assert (" + self.logicOperator + " " + str(
            self.df.columns.values[self.feIndex] + str(0)) + " " + str(self.feValue) + "))"
        if (self.logicOperator == 'not(='):
            assumption += ')'
        assumption += '\n'
        self.save_assumptions(assumption)

    def visit_expr5(self, node, children):
        temp_expr = node.text
        self.replaceIndex(temp_expr)
        assumption = '\n'
        assumption += "(assert (" + self.logicOperator + " " + str(
            self.df.columns.values[self.feIndex] + str(0)) + " " + str(
            self.df.columns.values[self.feIndex] + str(1)) + "))"
        if (self.logicOperator == 'not(='):
            assumption += ')'
        assumption += '\n'
        self.save_assumptions(assumption)

    def visit_expr6(self, node, children):
        if (self.arrFlag == True):
            self.trojan_expr(node, children)
        else:
            temp_expr = node.text
            self.replaceIndex(temp_expr)
            assumption = '\n'
            assumption += "(assert (" + self.logicOperator + " " + str(
                self.df.columns.values[self.feIndex] + str(0)) + " " + str(
                self.df.columns.values[self.feIndex] + str(1)) + "))"
            if (self.logicOperator == 'not(='):
                assumption += ')'
            assumption += '\n'
            self.save_assumptions(assumption)

    def visit_expr7(self, node, children):
        self.varMapDict['no_assumption'] = 'True'
        self.storeMapping()

    def trojan_expr(self, node, children):
        assumption = '\n'
        assumption += "(assert (" + self.logicOperator + " " + str(
            self.df.columns.values[self.feIndex] + str(0)) + " " + str(self.valueArr[self.feIndex]) + "))"
        if (self.logicOperator == 'not(='):
            assumption += ')'
        assumption += '\n'
        self.save_assumptions(assumption)
        self.storeMapping()

    def replaceIndex(self, temp_expr):
        for i in range(0, len(self.classVarList)):
            self.varList.append(self.classVarList[i].split('[', 1)[0])

        for i in range(0, len(self.classVarList)):
            if (self.classVarList[i] in temp_expr):
                temp_expr = temp_expr.replace(str(self.classVarList[i]), str(self.df.columns.values[self.feIndex]
                                                                             + str(i)))
                for j in range(0, len(self.varList)):
                    if (self.varList[j] in self.classVarList[i]):
                        self.varMapDict[self.varList[j]] = str(self.df.columns.values[self.feIndex] + str(i))
                        self.feArr.append(self.df.columns.values[self.feIndex] + str(i))
        self.varMapDict['no_assumption'] = False
        self.storeMapping()

    def getMapping(self):
        return self.varMapDict

    def storeInd(self, index):
        self.varInd = True
        self.feIndex = index
        self.checkValFeIndex()

    def storeArr(self, arr):
        self.valueArr = arr
        self.arrFlag = True

    def getPrefixExp(self, temp_expr1):
        for i in range(0, len(self.classVarList)):
            self.varList.append(self.classVarList[i].split('[', 1)[0])

        for i in range(0, len(self.classVarList)):
            if (self.classVarList[i] in temp_expr1):
                temp_expr1 = temp_expr1.replace(str(self.classVarList[i]), str(self.df.columns.values[self.feIndex]
                                                                               + str(i)))
                for j in range(0, len(self.varList)):
                    if (self.varList[j] in self.classVarList[i]):
                        self.varMapDict[self.varList[j]] = str(self.df.columns.values[self.feIndex] + str(i))
                        self.feArr.append(self.df.columns.values[self.feIndex] + str(i))

        self.varMapDict['no_assumption'] = False
        self.storeMapping()
        prefix_obj = InfixConverter()
        prefix_expr = prefix_obj.convert(temp_expr1)
        self.prefix_list = utils.String2List(prefix_expr)

    def storeMapping(self):
        if (self.arrFlag == True):
            self.varMapDict['no_mapping'] = 'True'
            self.varMapDict['no_assumption'] = 'False'
        else:
            self.varMapDict['no_mapping'] = 'False'
        self.final_var_mapping.append(self.varMapDict)

    def expr2logic(self, prefix_list):
        abs_flag = False
        assumption = '\n \n'
        assumption += "(assert (" + self.logicOperator + " "
        count_par = 2
        if (self.logicOperator == 'not(='):
            count_par += 1

        for el in prefix_list:
            for op in self.arithOperator:
                if (el == 'abs'):
                    abs_flag = True
                    for i in range(0, self.df.shape[1]):
                        temp = str(self.df.columns.values[i])
                        if (temp in self.feArr[0]):
                            feature = self.df.columns.values[i]
                            break
                    if ('float' in str(self.df.dtypes[feature])):
                        assumption += "(" + 'absoluteReal' + " "
                    else:
                        assumption += "(" + 'absoluteInt' + " "
                    count_par += 1
                    break
                if (op in el):
                    assumption += "(" + op + " "
                    count_par += 1
            for op in self.numList:
                if (op in el):
                    assumption += op + " "
            for op in self.feArr:
                if (op in el):
                    assumption += op + " "
            if (el == ')'):
                if (abs_flag == True):
                    count_par -= 2
                    assumption += '))'
                else:
                    count_par -= 1
                    assumption += ')'
        if (len(self.arithOperator) > 1):
            assumption += ") "
            count_par -= 1
        assumption += self.numEnd
        while (count_par >= 1):
            assumption += ")"
            count_par -= 1

        assumption += '\n'
        self.save_assumptions(assumption)

    def checkValFeIndex(self):
        if (self.feIndex > self.noOfAttr):
            raise Exception("Feature Index exceed maximum no. Of features in the data")

    def checkIndexConstncy(self):
        digit1 = int(re.search(r'\d+', self.classVarList[0]).group(0))
        if (len(self.classVarList) > 1):
            digit2 = int(re.search(r'\d+', self.classVarList[1]).group(0))
            if (digit1 != digit2):
                raise Exception("Feature Indexes don't match")


def functrainDecTree(oracle_data_file_path, outcome_name):
    df = pd.read_csv(oracle_data_file_path)
    X = df.drop(columns=[outcome_name])
    Y = df[outcome_name]
    model = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=20, min_samples_split=2,
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=0)

    model = model.fit(X, Y)
    dump(model, '../Model/multi_label_tree.joblib')

    return model


class RunChecker:
    def __init__(self, oracle_data_file_path, assumptions_file_path, assertions_file_path, params, params_dict):
        self.assumptions_file_path = assumptions_file_path
        self.assertions_file_path = assertions_file_path
        self.constraints_file_path = '../consmt.smt'
        self.tree_output_file_path = '../TreeOutput.txt'
        self.final_output_file_path = '../FinalOutput.txt'
        self.z3_df_output_file_path = '../TestDataSMT.csv'
        self.candidates_file_path = '../CandidateSet.csv'
        self.test_smt_file_path = '../TestDataSMTMain.csv'
        self.candidate_set_inst_file_path = '../CandidateSetInst.csv'
        self.candidate_set_branch_file_path = '../CandidateSetBranch.csv'
        self.test_set_file_path = '../TestSet.csv'
        self.cex_set_file_path = '../CexSet.csv'
        self.cand_set_file_path = '../Cand-set.csv'
        self.testing_data_file_path = '../TestingData.csv'
        self.dnn_smt_file_path = 'DNNSmt.smt2'
        self.dec_smt_file_path = '../DecSmt.smt2'
        self.toggle_feature_smt_file_path = '../ToggleFeatureSmt.smt2'
        self.toggle_branch_smt_file_path = '../ToggleBranchSmt.smt2'
        self.condition_file_path = '../ConditionFile.txt'
        self.oracle_data_file_path = oracle_data_file_path
        self.oracle_data = pd.read_csv(self.oracle_data_file_path)
        self.params = params

        self.delete_files([self.constraints_file_path, self.tree_output_file_path, self.final_output_file_path,
                           self.z3_df_output_file_path, self.candidates_file_path, self.test_smt_file_path,
                           self.candidate_set_inst_file_path, self.candidate_set_branch_file_path,
                           self.test_set_file_path, self.cex_set_file_path, self.cand_set_file_path,
                           self.testing_data_file_path, self.dnn_smt_file_path, self.dec_smt_file_path,
                           self.toggle_branch_smt_file_path, self.toggle_feature_smt_file_path,
                           self.condition_file_path])
        self.paramDict = params_dict
        if not self.params['MUTcontent']:
            self.model_type = self.paramDict['model_type']
            if 'model_path' in self.paramDict:
                model_path = self.paramDict['model_path']
                if self.model_type == 'Pytorch':
                    self.model = NetArch1()
                    self.model = torch.load(model_path)
                    self.model.eval()
                else:
                    self.model = load(model_path)
            else:
                self.model = load('Model/MUT.joblib')
        else:
            dfWeight = pd.read_csv('MUTWeight.csv')
            pred_weight = dfWeight.values
            pred_weight = pred_weight[:, :-1]
            self.model = pred_weight
        with open(self.test_set_file_path, 'w', newline='') as csvfile:
            fieldnames = self.oracle_data.columns.values
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
        with open(self.cex_set_file_path, 'w', newline='') as csvfile:
            fieldnames = self.oracle_data.columns.values
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)

    def delete_files(self, file_paths):
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"{file_path} has been deleted.")
            else:
                print(f"{file_path} does not exist.")

    def funcCreateOracle(self):
        dfTest = pd.read_csv(self.testing_data_file_path)
        data = dfTest.values
        predict_class = self.model.predict(data)
        for i in range(0, data.shape[0]):
            dfTest.loc[i, 'Class'] = predict_class[i]
        dfTest.to_csv(self.oracle_data_file_path, index=False, header=True)

    def chkPairBel(self, tempMatrix, noAttr):
        firstTest = np.zeros((noAttr,))
        secTest = np.zeros((noAttr,))
        dfT = pd.read_csv('TestingSet.csv')
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
        dfTest = pd.read_csv('TestingSet.csv')
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
        dfTr = pd.read_csv('../Datasets/mnist_resized.csv')
        noOfAttr = dfTr.shape[1] - 1
        i = 0
        for lines in self.assumptions_file_path:
            if p == '\n':
                pass
            else:
                for col in dfTr.columns.values:
                    if col in lines:
                        num = float(re.search(r'[+-]?([0-9]*[.])[0-9]+', lines).group(0))
                        assume_dict.append(num)
                        i += 1
        num1 = float(re.search(r'[+-]?([0-9]*[.])[0-9]+', self.assumptions_file_path).group(0))
        with open('TestingSet.csv', 'w', newline='') as csvfile:
            fieldnames = dfTr.columns.values
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
        dfAg = pd.read_csv('TestingSet.csv')
        dfAg.drop('Class', axis=1, inplace=True)
        dfAg.to_csv('TestingSet.csv', index=False, header=True)
        inst_count = 0
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
                with open('TestingSet.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(tempMatrix)
            inst_count = inst_count + 1
        cexPair, flag = self.chkAttack(num1)
        if flag:
            return cexPair, True
        return cexPair, False

    def funcPrediction(self, X, dfCand, testIndx):
        if self.params['MUTcontent']:
            if self.model_type == 'Pytorch':
                X_pred = torch.tensor(X[testIndx], dtype=torch.float32)

                predict_prob = self.model(X_pred.view(-1, X.shape[1]))
                return int(torch.argmax(predict_prob))
            else:
                if self.params['MUTcontent']:
                    return self.model.predict(utils.convDataInst(X, dfCand, self.params, testIndx, 1))
        else:
            temp_class = self.model.predict(utils.convDataInst(X, dfCand, self.paramDict, testIndx, 1))[0]
            if temp_class < 0:
                return 0
            else:
                return temp_class

    def addModelPred(self):
        dfCexSet = pd.read_csv(self.cex_set_file_path)
        dataCex = dfCexSet.values
        if self.params['MUTcontent']:
            if self.model_type == 'Pytorch':
                X = dataCex[:, :-1]
                X = torch.tensor(X, dtype=torch.float32)
                predict_class = []
                for i in range(0, X.shape[0]):
                    predict_prob = self.model(X[i].view(-1, X.shape[1]))
                    predict_class.append(int(torch.argmax(predict_prob)))
            else:
                predict_class = self.model.predict(dataCex[:, :-1])
            for i in range(0, dfCexSet.shape[0]):
                dfCexSet.loc[i, 'Class'] = predict_class[i]
        else:
            predict_class = self.model.predict(dataCex[:, :-1])
            for i in range(0, dfCexSet.shape[0]):
                dfCexSet.loc[i, 'Class'] = predict_class[i]
        dfCexSet.to_csv(self.cex_set_file_path, index=False, header=True)

    def runWithDNN(self):
        self.no_of_params = int(self.paramDict['no_of_params'])
        retrain_flag = False
        MAX_CAND_ZERO = 10
        count_cand_zero = 0
        count = 0
        start_time = time.time()
        if self.no_of_params == 1:
            cex, ret_flag = self.checkWithOracle()
            if ret_flag:
                with open(self.cex_set_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(np.reshape(np.array(cex), (1, self.oracle_data.shape[1] - 1)))
                self.addModelPred()
                return 0

        while count < self.max_samples:
            functrainDNN()
            print('count is:', count)
            obj_dnl = ConvertDNN2logic()
            obj_dnl.funcDNN2logic()
            utils.storeAssumeAssert('DNNSmt.smt2')
            utils.addSatOpt('DNNSmt.smt2')
            os.system(r"z3 DNNSmt.smt2 > FinalOutput.txt")
            satFlag = utils.funcConvZ3OutToData(self.oracle_data)
            if not satFlag:
                if count == 0:
                    print('No CEX is found by the checker in the first trial')
                    return 0
                elif (count != 0) and (self.mul_cex == 'True'):
                    dfCexSet = pd.read_csv(self.cex_set_file_path)
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
                funcAddCex2CandidateSet()
                funcAddCexPruneCandidateSet4DNN()
                funcCheckCex()
                # Increase the count if no further candidate cex has been found
                dfCand = pd.read_csv(self.cand_set_file_path)
                if round(dfCand.shape[0] / self.no_of_params) == 0:
                    count_cand_zero += 1
                    if count_cand_zero == MAX_CAND_ZERO:
                        if self.mul_cex == 'True':
                            dfCexSet = pd.read_csv(self.cex_set_file_path)
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
                        if self.mul_cex == 'True':
                            with open(self.cex_set_file_path, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(temp_store)
                        else:
                            print('A counter example is found, check it in CexSet.csv file: ', temp_store)
                            with open(self.cex_set_file_path, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(temp_store)
                            self.addModelPred()
                            return 1
                    else:
                        utils.funcAdd2Oracle(temp_add_oracle)

                    if retrain_flag:
                        self.funcCreateOracle()

                if (time.time() - start_time) > self.deadline:
                    print("Time out")
                    break

        dfCexSet = pd.read_csv(self.cex_set_file_path)
        if (round(dfCexSet.shape[0] / self.no_of_params) > 0) and (count >= self.max_samples):
            self.addModelPred()
            print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
            print('No. of Samples looked for counter example has exceeded the max_samples limit')
        else:
            print('No counter example has been found')

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
            tree_model = functrainDecTree(self.oracle_data_file_path, self.params['output_class'])
            functree2LogicMain(tree_model, self.oracle_data_file_path, self.no_of_params, self.params['output_class'],
                               tree_output_file=self.tree_output_file_path, smt_file=self.dec_smt_file_path,
                               constraint_file=self.constraints_file_path)
            utils.storeAssumeAssert(self.constraints_file_path, self.params, self.assumptions_file_path,
                                    self.assertions_file_path)
            utils.addSatOpt(self.constraints_file_path)
            os.system(f"z3 {self.constraints_file_path} > {self.final_output_file_path}")
            satFlag = utils.funcConvZ3OutToData(self.oracle_data, self.paramDict,
                                                self.final_output_file_path, self.z3_df_output_file_path)
            if not satFlag:
                if count == 0:
                    print('No CEX is found by the checker at the first trial')
                    return 0
                elif (count != 0) and (self.mul_cex == 'True'):
                    dfCexSet = pd.read_csv(self.cex_set_file_path)
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
                funcAddCex2CandidateSet(self.z3_df_output_file_path, self.candidates_file_path)
                funcAddCexPruneCandidateSet(tree_model, self.paramDict, self.z3_df_output_file_path,
                                            self.test_smt_file_path, self.tree_output_file_path,
                                            self.candidate_set_inst_file_path, self.candidate_set_branch_file_path,
                                            self.candidates_file_path, self.final_output_file_path,
                                            self.dnn_smt_file_path, self.dec_smt_file_path,
                                            self.toggle_feature_smt_file_path, self.toggle_branch_smt_file_path,
                                            self.condition_file_path)
                funcCheckCex(self.candidates_file_path, self.test_set_file_path, self.cand_set_file_path)
                dfCand = pd.read_csv(self.cand_set_file_path)
                if round(dfCand.shape[0] / self.no_of_params) == 0:
                    count_cand_zero += 1
                    if count_cand_zero == MAX_CAND_ZERO:
                        if self.mul_cex == 'True':
                            dfCexSet = pd.read_csv(self.cex_set_file_path)
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
                        if self.mul_cex == 'True':
                            with open(self.cex_set_file_path, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(temp_store)
                        else:
                            print('A counter example is found, check it in CexSet.csv file: ', temp_store)
                            with open(self.cex_set_file_path, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(temp_store)
                            self.addModelPred()
                            return 1
                    else:
                        utils.funcAdd2Oracle(temp_add_oracle, self.testing_data_file_path)

                if retrain_flag:
                    self.funcCreateOracle()

                if (time.time() - start_time) > self.deadline:
                    print("Time out")
                    break

        dfCexSet = pd.read_csv(self.cex_set_file_path)
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

        if white_box == 'DNN':
            self.runWithDNN()
        else:
            self.runWithTree()


class AssertionVisitor(NodeVisitor):

    def __init__(self, oracle_data_df, params, params_dict, filepath='asrt.smt'):
        self.currentClass = []
        self.modelVarList = []
        self.classNameList = []
        self.currentOperator = ""
        self.negOp = ""
        self.varList = []
        self.mydict = {}
        self.varMap = {}
        self.feVal = 0
        self.count = 0
        self.dfOracle = oracle_data_df
        self.mydict = params
        self.paramDict = params_dict
        self.filepath = filepath

    def save_assertions(self, assertion):
        with open(self.filepath, 'a') as file:
            file.write(assertion)

    def generic_visit(self, node, children):
        pass

    def visit_classVar(self, node, children):
        if self.mydict['no_mapping']:
            pass
        else:
            for el in self.varList:
                if el in node.text:
                    if self.mydict['no_assumption']:
                        className = 'Class' + str(self.mydict[el])
                    else:
                        className = 'Class' + str(self.count - 1)
            self.currentClass.append(className)

    def visit_neg(self, node, children):
        self.negOp = node.text

    def visit_model_name(self, node, children):
        self.modelVarList.append(node.text)

    def visit_class_name(self, node, children):
        if (node.text in self.dfOracle.columns.values):
            self.classNameList.append(node.text)
        else:
            raise Exception('Class name ' + str(node.text) + ' do not exist')

    def visit_variable(self, node, children):
        if (self.mydict['no_mapping'] == 'True'):
            pass
        else:
            self.varList.append(node.text)
            if (self.mydict['no_assumption'] == 'False'):
                num = str(int(re.search(r'\d+', self.mydict[node.text]).group(0)))
                self.mydict[node.text] = num[len(num) - 1]
            else:
                if (node.text in self.varMap):
                    pass
                else:
                    self.varMap[node.text] = self.count
                    self.count += 1

    def visit_operator(self, node, children):
        if ('!=' in node.text):
            self.currentOperator = 'not(= '
        elif ('==' in node.text):
            self.currentOperator = '= '
        else:
            self.currentOperator = node.text

    def visit_number(self, node, children):
        self.feVal = float(node.text)

    def visit_expr1(self, node, children):
        if (self.mydict['no_mapping'] == 'True'):
            assertStmnt = ('(assert(not (', self.currentOperator, ' Class', str(0), ' ', str(self.feVal), ')))')
        else:
            assertStmnt = ('(assert(not (', self.currentOperator, self.currentClass[0], ' ', str(self.feVal), ')))')
        assertion = ''.join(assertStmnt)
        if (self.currentOperator == 'not(= '):
            assertion += ')'
        self.save_assertions(assertion)

    def checkModelName(self):
        if (self.modelVarList[0] != self.modelVarList[1]):
            raise Exception('Model names do not match')

    def visit_expr2(self, node, children):
        assertStmnt = ('(assert(not (', self.currentOperator, self.currentClass[0], ' ', self.currentClass[1], ')))')
        assertion = '\n' + ''.join(assertStmnt)
        if (self.currentOperator == 'not(= '):
            assertion += ')'
        self.save_assertions(assertion)

    def visit_expr3(self, node, children):
        if (self.count > int(self.paramDict['no_of_params'])):
            raise Exception('The no. of parameters mentioned exceeded in assert statement')

        self.checkModelName()
        if (self.negOp == '~'):
            if (self.paramDict['white_box_model'] == 'DNN'):
                assertStmnt = (
                    '(assert(not (', self.currentOperator, ' (= ', self.classNameList[0], str(self.count - 1),
                    ' 1)', ' (not ', ' (= ', self.classNameList[1], str(self.count - 1), ' 1)', '))))')
            else:
                assertStmnt = ('(assert(not (', self.currentOperator, ' (= ', self.classNameList[0], ' 1)',
                               ' (not ', ' (= ', self.classNameList[1], ' 1)', '))))')
        else:
            if (self.paramDict['white_box_model'] == 'DNN'):
                assertStmnt = (
                    '(assert(not (', self.currentOperator, ' (= ', self.classNameList[0], str(self.count - 1), ' 1)',
                    ' ',
                    ' (= ', self.classNameList[1], str(self.count - 1), ' 1)', ')))')
            else:
                assertStmnt = ('(assert(not (', self.currentOperator, ' (= ', self.classNameList[0], ' 1)', ' ',
                               ' (= ', self.classNameList[1], ' 1)', ')))')
        assertion = '\n' + ''.join(assertStmnt)
        self.save_assertions(assertion)

    def checkFeConsist(self):
        if (len(self.varList) == len(self.mydict) - 2):
            for el in self.varList:
                if (el not in self.mydict.keys()):
                    raise Exception("Unknown feature vector")

        else:
            raise Exception("No. of feature vectors do not match with the assumption")


def Assume(oracle_data_file_path, file_path='ams.smt', *args):
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

    df = pd.read_csv(oracle_data_file_path)

    tree = grammar.parse(args[0])
    assumeVisitObj = AssumptionVisitor(df, file_path)
    if len(args) == 3:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.storeArr(args[2])
        assumeVisitObj.visit(tree)
    elif len(args) == 2:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.visit(tree)
    elif len(args) == 1:
        assumeVisitObj.visit(tree)

    return assumeVisitObj.final_var_mapping


def Assert(oracle_data_file_path, params, prop_check_params, file_path='asrt.smt', *args):
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

    oracle_data = pd.read_csv(oracle_data_file_path)
    tree = grammar.parse(args[0])
    assertVisitObj = AssertionVisitor(oracle_data, params, prop_check_params, filepath=file_path)
    assertVisitObj.visit(tree)
