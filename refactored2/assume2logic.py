import pandas as pd
import csv as cv
from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
import re
from paths import HERE
from refactored2 import util
import sys

from refactored2.util import local_save


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

    '''
    def isOperand(self, op):
        for var in self.varList:
            if(op == var):
                return var
            else:
                return op.isdigit()
    '''

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
        # print(output)
        return output

    '''
     1. Reverse expression string
     2. Replace open paren with close paren and vice versa
     3. Get Postfix and reverse it
    '''

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
            result = eval(expr);
        except:
            result = expr
        return self.toPrefix(expr)


# In[129]:


class AssumptionVisitor(NodeVisitor):

    def __init__(self):
        self.varList = []
        self.classVarList = []
        self.arithOperator = []
        self.logicOperator = ""
        self.numList = []
        self.numEnd = ""
        self.feIndex = 99999
        self.feValue = 0
        self.count = 0
        self.df = pd.read_csv('files/OracleData.csv')
        self.feArr = []
        self.noOfAttr = self.df.shape[1]
        self.varMapDict = {}
        self.prefix_list = []
        self.varInd = False
        self.arrFlag = False

    def generic_visit(self, node, children):
        pass

    def visit_arith_op(self, node, children):
        self.arithOperator.append(node.text)

    def visit_logic_op(self, node, children):
        if '!=' in node.text:
            self.logicOperator = 'not(='
        else:
            self.logicOperator = node.text

    def visit_number(self, node, children):
        self.numList.append(node.text)

    def visit_classVar(self, node, children):
        if self.varInd:
            raise Exception("Feature indexes given twice")
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
        assume_stmnt = "(assert (" + self.logicOperator + " " + str(self.df.columns.values[self.feIndex] + str(0)) + " " + str(self.feValue) + "))"
        if self.logicOperator == 'not(=':
            assume_stmnt += ')'
        local_save(assume_stmnt, 'assumeStmnt', force_rewrite=False)

    def visit_expr4(self, node, children):
        temp_expr = node.text
        self.replaceIndex(temp_expr)
        assume_stmnt = "(assert (" + self.logicOperator + " " + str(self.df.columns.values[self.feIndex] + str(0)) + " " + str(self.feValue) + "))"
        if self.logicOperator == 'not(=':
            assume_stmnt += ')'
        local_save(assume_stmnt, 'assumeStmnt', force_rewrite=False)

    def visit_expr5(self, node, children):
        temp_expr = node.text
        self.replaceIndex(temp_expr)
        assume_stmnt = "(assert (" + self.logicOperator + " " + str(self.df.columns.values[self.feIndex] + str(0)) + " " + str(self.df.columns.values[self.feIndex] + str(1)) + "))"
        if self.logicOperator == 'not(=':
            assume_stmnt += ')'
        local_save(assume_stmnt, 'assumeStmnt', force_rewrite=False)

    def visit_expr6(self, node, children):
        if self.arrFlag:
            self.trojan_expr(node, children)
        else:
            temp_expr = node.text
            self.replaceIndex(temp_expr)
            assume_stmnt = "(assert (" + self.logicOperator + " " + str(self.df.columns.values[self.feIndex] + str(0)) + " " + str(self.df.columns.values[self.feIndex] + str(1)) + "))"
            if self.logicOperator == 'not(=':
                assume_stmnt += ')'
            local_save(assume_stmnt, 'assumeStmnt', force_rewrite=False)

    def visit_expr7(self, node, children):
        self.varMapDict['no_assumption'] = True
        self.storeMapping()

    def trojan_expr(self, node, children):
        assume_stmnt = "(assert (" + self.logicOperator + " " + str(self.df.columns.values[self.feIndex] + str(0)) + " " + str(self.valueArr[self.feIndex]) + "))"
        if self.logicOperator == 'not(=':
            assume_stmnt += ')'
        local_save(assume_stmnt, 'assumeStmnt', force_rewrite=False)
        self.storeMapping()

    # Later use this function to convert statement in the assume to prefix expression form
    def replaceIndex(self, temp_expr):
        for i in range(len(self.classVarList)):
            self.varList.append(self.classVarList[i].split('[', 1)[0])

        for i in range(len(self.classVarList)):
            if self.classVarList[i] in temp_expr:
                temp_expr = temp_expr.replace(str(self.classVarList[i]), str(self.df.columns.values[self.feIndex] + str(i)))
                for j in range(len(self.varList)):
                    if self.varList[j] in self.classVarList[i]:
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
        for i in range(len(self.classVarList)):
            self.varList.append(self.classVarList[i].split('[', 1)[0])

        for i in range(len(self.classVarList)):
            if self.classVarList[i] in temp_expr1:
                temp_expr1 = temp_expr1.replace(str(self.classVarList[i]), str(self.df.columns.values[self.feIndex] + str(i)))
                for j in range(len(self.varList)):
                    if self.varList[j] in self.classVarList[i]:
                        self.varMapDict[self.varList[j]] = str(self.df.columns.values[self.feIndex] + str(i))
                        self.feArr.append(self.df.columns.values[self.feIndex] + str(i))

        self.varMapDict['no_assumption'] = False
        self.storeMapping()
        prefix_obj = InfixConverter()
        prefix_expr = prefix_obj.convert(temp_expr1)
        self.prefix_list = list(prefix_expr.split(" "))

    def storeMapping(self):
        if self.arrFlag:
            self.varMapDict['no_mapping'] = True
            self.varMapDict['no_assumption'] = False
        else:
            self.varMapDict['no_mapping'] = False
        try:
            local_save(self.varMapDict, 'dict', force_rewrite=True)
        except IOError:
            print("I/O error")

    def expr2logic(self, prefix_list):
        abs_flag = False
        logic_stmnt = "(assert (" + self.logicOperator + " "
        count_par = 2
        if self.logicOperator == 'not(=':
            count_par += 1

        for el in prefix_list:
            for op in self.arithOperator:
                if el == 'abs':
                    abs_flag = True
                    for i in range(self.df.shape[1]):
                        temp = str(self.df.columns.values[i])
                        if temp in self.feArr[0]:
                            feature = self.df.columns.values[i]
                            break
                    if 'float' in str(self.df.dtypes[feature]):
                        logic_stmnt += "(" + 'absoluteReal' + " "
                    else:
                        logic_stmnt += "(" + 'absoluteInt' + " "
                    count_par += 1
                    break
                if op in el:
                    logic_stmnt += "(" + op + " "
                    count_par += 1
            for op in self.numList:
                if op in el:
                    logic_stmnt += op + " "
            for op in self.feArr:
                if op in el:
                    logic_stmnt += op + " "
            if el == ')':
                if abs_flag:
                    count_par -= 2
                    logic_stmnt += '))'
                else:
                    count_par -= 1
                    logic_stmnt += ')'
        if len(self.arithOperator) > 1:
            logic_stmnt += ") "
            count_par -= 1
        logic_stmnt += self.numEnd
        while count_par >= 1:
            logic_stmnt += ")"
            count_par -= 1

        local_save(logic_stmnt, 'assumeStmnt', force_rewrite=False)

    def checkValFeIndex(self):
        if self.feIndex > self.noOfAttr:
            raise Exception("Feature Index exceed maximum number of features in the data")

    def checkIndexConstncy(self):
        digit1 = int(re.search(r'\d+', self.classVarList[0]).group(0))
        if len(self.classVarList) > 1:
            digit2 = int(re.search(r'\d+', self.classVarList[1]).group(0))
            if digit1 != digit2:
                raise Exception("Feature Indexes don't match")


def Assume(*args):
    grammar = Grammar(
        r"""

    expr        = expr1 / expr2 / expr3 /expr4 /expr5 / expr6
    expr1       = expr_dist1 logic_op num_log
    expr2       = expr_dist2 logic_op num_log
    expr3       = classVar ws logic_op ws value
    expr4       = classVarArr ws logic_op ws value
    expr5       = classVar ws logic_op ws classVar
    expr6       = classVarArr ws logic_op ws classVarArr
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
    assumeVisitObj = AssumptionVisitor()
    if (len(args) == 2):
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.visit(tree)
    else:
        assumeVisitObj.visit(tree)
