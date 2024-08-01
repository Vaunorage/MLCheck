import pandas as pd
import csv as cv
import numpy as np
from refactored2 import util
import os
import re

from refactored2.util import local_load, local_save


def convert_z3_output_to_df(file_content, df, paramDict):

    # Regex pattern to match the define-fun declarations
    pattern = r'\(define-fun\s+(\w+?)(\d)\s+\(\)\s+(\w+)\s*\n\s*(\d+)\)'

    # Find all matches
    matches = re.findall(pattern, file_content)

    # Initialize an empty DataFrame to store results
    no_of_params = int(paramDict['no_of_params'])
    dfAgain = pd.DataFrame(np.zeros((no_of_params, df.shape[1])), columns=df.columns.values)

    # Process each match
    for feature_name, param_no, data_type, value in matches:

        param_no = int(param_no)
        value = int(value)  # assuming the value is always an integer here

        # Assign the value to the correct cell in the DataFrame
        if feature_name in df.columns:
            dfAgain.loc[param_no, feature_name] = value

    return dfAgain


def funcConvZ3OutToData(df):
    paramDict = local_load('param_dict')
    # testMatrix = np.zeros(((no_of_params), df.shape[1]))

    # if (os.stat('files/FinalOutput.txt').st_size > 0):
    #     with open('files/FinalOutput.txt') as f1:

    file_content = local_load('FinalOutput')

    if ('unknown' in file_content[0]):
        raise Exception('Encoding problem')
    if ('model is not available' in file_content[1]):
        return False

    # Create DataFrame from the data dictionary
    dfAgain = convert_z3_output_to_df(file_content, df, paramDict)

    # dfAgain.to_csv('files/TestDataSMT.csv', index=False, header=True)
    local_save(dfAgain, 'TestDataSMT', force_rewrite=True)
    return True

    # else:
    #     raise Exception("There is no solver installed in your system")
