import pandas as pd
import csv as cv
import numpy as np
import os
import re


def String2List(string):
    return list(string.split(" "))


def file_len(fname):
    if (os.stat(fname).st_size == 0):
        return 'empty'
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def convDataInst(X, df, params, j, no_of_class):
    if (params['multi_label'] == 'False'):
        no_of_class = 1
    data_inst = np.zeros((1, df.shape[1] - no_of_class))
    if (j > X.shape[0]):
        raise Exception('Z3 has produced counter example with all 0 values of the features: Run the script Again')
    for i in range(df.shape[1] - no_of_class):
        data_inst[0][i] = X[j][i]
    return data_inst


def funcAdd2Oracle(data, testing_file_path='TestingData.csv'):
    with open(testing_file_path, 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(data)


def funcCreateOracle(no_of_class, multi_label, model, testing_file_path='TestingData.csv'):
    df = pd.read_csv(testing_file_path)
    data = df.values
    if multi_label == 'False':
        X = data[:, :-1]
        predict_class = model.predict(X)
        for i in range(0, X.shape[0]):
            df.loc[i, 'Class'] = predict_class[i]
    else:
        X = data[:, :-no_of_class]
        predict_class = model.predict(X)
        index = df.shape[1] - no_of_class
        for i in range(0, no_of_class):
            className = str(df.columns.values[index + i])
            for j in range(0, X.shape[0]):
                df.loc[j, className] = predict_class[j][i]
    df.to_csv('OracleData.csv', index=False, header=True)


def storeMapping(file_name, dictionary):
    try:
        with open(file_name, 'w') as csv_file:
            writer = cv.writer(csv_file)
            for key, value in dictionary.items():
                writer.writerow([key, value])
    except IOError:
        print("I/O error")


def addContent(file_name, f_content):
    f1 = open(file_name, 'a')
    for x in f_content:
        f1.write('\n')
        f1.write(x)
    f1.close()


def addSatOpt(constraint_file_path):
    with open(constraint_file_path, 'a') as f:
        f.write('\n (check-sat) \n (get-model) \n')


def storeAssumeAssert(constrains_file_path, params, assumptions_file_path, assertions_file_path):
    # Read assumptions and append to constraints
    if not params['no_assumption']:
        with open(assumptions_file_path, 'r') as assumptions_file:
            assumptions_content = assumptions_file.read()

        with open(constrains_file_path, 'a') as constraints_file:
            constraints_file.write(assumptions_content)

    # Read assertions and append to constraints
    with open(assertions_file_path, 'r') as assertions_file:
        assertions_content = assertions_file.read()

    with open(constrains_file_path, 'a') as constraints_file:
        constraints_file.write(assertions_content)


def funcConvZ3OutToData(df, paramDict, final_output_file_path='FinalOutput.txt',
                        z3_df_output_file_path='TestDataSMT.csv'):
    no_of_params = int(paramDict['no_of_params'])
    testMatrix = np.zeros(((no_of_params), df.shape[1]))

    if (os.stat(final_output_file_path).st_size > 0):
        with open(final_output_file_path) as f1:
            file_content = f1.readlines()

        file_content = [x.strip() for x in file_content]
        noOfLines = file_len(final_output_file_path)

        with open(z3_df_output_file_path, 'w', newline='') as csvfile:
            fieldnames = df.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
            writer.writerows(testMatrix)

        dfAgain = pd.read_csv(z3_df_output_file_path)
        if ('unknown' in file_content[0]):
            raise Exception('Encoding problem')
        if ('model is not available' in file_content[1]):
            return False
        else:
            i = 1
            while (i < noOfLines):
                minus_flag = False
                fe_flag = False
                if ("(model" == file_content[i]):
                    i = i + 1
                elif (")" == file_content[i]):
                    i = i + 1
                else:
                    for j in range(0, df.columns.values.shape[0]):
                        for param_no in range(0, no_of_params):
                            if (paramDict['multi_label'] == 'True' and no_of_params == 1 \
                                    and paramDict['white_box_model'] == 'Decision tree'):
                                fe_add = ' '
                            else:
                                fe_add = str(param_no)
                            if (df.columns.values[j] + fe_add in file_content[i]):
                                feature_name = df.columns.values[j]
                                fe_flag = True
                                if ('Int' in file_content[i]):
                                    i = i + 1
                                    digit = int(re.search(r'\d+', file_content[i]).group(0))
                                    if ('-' in file_content[i]):
                                        digit = 0 - digit
                                elif ('Real' in file_content[i]):
                                    i = i + 1
                                    if ("(/" in file_content[i]):
                                        if ('-' in file_content[i]):
                                            minus_flag = True
                                        multi_digits = re.findall('\d*?\.\d+', file_content[i])
                                        if (len(multi_digits) == 1):
                                            i = i + 1
                                            multi_digits.append(float(re.search(r'\d+', file_content[i]).group(0)))
                                        digit = float(multi_digits[0]) / float(multi_digits[1])
                                        if (minus_flag == True):
                                            digit = 0 - digit
                                    else:
                                        digit = float(re.search(r'\d+', file_content[i]).group(0))
                                        if ('-' in file_content[i]):
                                            digit = 0 - digit
                                    dfAgain.loc[param_no, feature_name] = digit
                                i = i + 1
                    if (fe_flag == False):
                        i = i + 2
            dfAgain.to_csv(z3_df_output_file_path, index=False, header=True)
            return True

    else:
        raise Exception("There is no solver installed in your system")
