from paths import HERE
from refactored2.mlCheck import Assume, Assert, propCheck
import statistics as st
import math

from refactored2.util import local_delete, local_load

iteration_no = 1


def delete_all():
    files = ['assumeStmnt', 'assertStmnt', 'Cand-Set', 'CandidateSet',
             'CandidateSetInst', 'CandidateSetBranch', 'TestDataSMT',
             'TestDataSMTMain', 'DecSmt', 'ToggleBranchSmt',
             'ToggleFeatureSmt', 'TreeOutput', 'SampleFile', 'FinalOutput',
             'MUTWeight', 'ConditionFile', 'MUTWeight', 'MUTWeight', 'DNNSmt',
             'TestData', 'TestDataSet', 'CandTestDataSet']
    for ff in files:
        local_delete(ff)


def func_calculate_sem(samples):
    standard_dev = st.pstdev(samples)
    return standard_dev / math.sqrt(len(samples))


delete_all()

model_path_list_adult = ['FairUnAwareTestCases/NBAdult.joblib']

white_box_list = ['Decision tree']

f = open('Output/fairnessResults', 'w')

for model_path in model_path_list_adult:

    f.write("Result of MLCheck is----- \n")
    for white_box in white_box_list:
        cex_count = 0
        cex_count_list = []
        cex_count_sem = 0
        if white_box == 'Decision tree':
            f.write('------MLC_DT results-----\n')
        for no in range(0, iteration_no):
            propCheck(no_of_params=2, max_samples=1500, model_type='sklearn', model_path=model_path, mul_cex=True,
                      train_data_available=True, train_ratio=30, no_of_train=1000,
                      train_data_loc=HERE.joinpath('refactored2/Datasets/Adult.csv').as_posix(),
                      white_box_model=white_box, no_of_class=2)

            for i in range(0, 13):
                if i == 8:
                    Assume('x[i] != y[i]', i)
                else:
                    Assume('x[i] = y[i]', i)
            Assert('model.predict(x) == model.predict(y)')
            dfCexSet = local_load('CexSet')
            cex_count = cex_count + round(dfCexSet.shape[0] / 2)
            cex_count_list.append(round(dfCexSet.shape[0] / 2))
        mean_cex_count = cex_count / iteration_no
        cex_count_sem = func_calculate_sem(cex_count_list)

        model_name = model_path.split('/')
        model_name = model_name[1].split('.')
        f.write('Result of ' + model_name[0] + ':\n')
        f.write('Mean value is: ' + str(mean_cex_count) + '\n')
        f.write('Standard Error of the Mean is: +- ' + str(cex_count_sem) + '\n \n ')
