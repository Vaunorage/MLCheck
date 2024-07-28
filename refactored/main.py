import os
import random
import time

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from refactored.assume2logic import Assume, Assert, RunChecker


class Feature:
    def __init__(self, name, type, min_val, max_val):
        self.name = name
        self.type = type
        self.min_val = min_val
        self.max_val = max_val


def parse_features(feature_data):
    features = []
    for feature in feature_data:
        features.append(Feature(
            name=feature['name'],
            type=feature['type'],
            min_val=feature['min_val'],
            max_val=feature['max_val']
        ))
    return features


class DataGenerator:
    def __init__(self, df, num_samples=1000, train_ratio=None):
        self.df = df
        self.num_samples = num_samples
        self.train_ratio = train_ratio
        self.features = self._infer_features()

    def _infer_features(self):
        features = []
        for column in self.df.columns:
            dtype = self.df[column].dtype
            if np.issubdtype(dtype, np.integer):
                feature_type = 'int'
                min_val = self.df[column].min()
                max_val = self.df[column].max()
            elif np.issubdtype(dtype, np.floating):
                feature_type = 'float'
                min_val = self.df[column].min()
                max_val = self.df[column].max()
            else:
                feature_type = 'category'
                min_val = None
                max_val = None

            features.append({
                'name': column,
                'type': feature_type,
                'min_val': min_val,
                'max_val': max_val,
                'categories': self.df[column].unique() if feature_type == 'category' else None
            })
        return features

    def generate_sample(self):
        sample = {}
        for feature in self.features:
            if feature['type'] == 'int':
                sample[feature['name']] = random.randint(feature['min_val'], feature['max_val'])
            elif feature['type'] == 'float':
                sample[feature['name']] = round(random.uniform(feature['min_val'], feature['max_val']), 2)
            else:  # category
                sample[feature['name']] = random.choice(feature['categories'])
        return sample

    def is_unique(self, samples, new_sample):
        return all(any(new_sample[f['name']] != s[f['name']] for f in self.features) for s in samples)

    def generate_data(self):
        samples = []
        while len(samples) < self.num_samples:
            new_sample = self.generate_sample()
            if self.is_unique(samples, new_sample):
                samples.append(new_sample)

        generated_df = pd.DataFrame(samples)

        if self.train_ratio is not None:
            train_samples = int(self.train_ratio * len(self.df))
            train_df = self.df.sample(n=train_samples)
            generated_df = pd.concat([generated_df, train_df], ignore_index=True)

        return generated_df


class OracleDataGenerator:
    def __init__(self, model):
        if not isinstance(model, BaseEstimator):
            raise ValueError("The provided model must be a scikit-learn estimator.")
        self.model = model

    def generate_oracle_data(self, input_data, y_class):
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("input_data must be a pandas DataFrame")

        X = input_data.drop(y_class, axis=1, errors='ignore')

        predictions = self.model.predict(X)

        # Add predictions to the input data
        result = input_data.copy()
        result[y_class] = predictions.astype(int)

        return result


class PropCheck:
    def __init__(self, model, input_data, params):
        self.model = self._load_model(model)
        self.input_data = input_data
        self.params = self._process_params(params)
        self.column_types = self.input_data.dtypes
        self.oracle_data_file_path = '../OracleData.csv'

    def _load_model(self, model):
        if isinstance(model, str):
            return load(model)
        elif isinstance(model, BaseEstimator):
            return model
        else:
            raise ValueError("Invalid sklearn model. Provide either a file path or a sklearn estimator.")

    def _process_params(self, params):
        default_params = {
            'num_samples': 1000,
            'train_ratio': None,
            'no_of_params': None,
            'deadlines': 500000,
            'white_box_model': 'Decision tree',
            'no_of_layers': 2,
            'layer_size': 64,
            'no_EPOCHS': 20,
            'mul_cex_opt': False,
            'max_samples': 1000,
            'no_of_class': 3
        }
        default_params.update(params)

        if default_params['no_of_params'] is None or default_params['no_of_params'] > 3:
            raise ValueError("Please provide a valid value for no_of_params (1-3)")

        return default_params

    def run(self):
        data_generator = DataGenerator(self.input_data, num_samples=self.params['num_samples'],
                                       train_ratio=self.params['train_ratio'])
        generated_data = data_generator.generate_data()

        oracle_generator = OracleDataGenerator(self.model)

        oracle_data = oracle_generator.generate_oracle_data(generated_data, self.params['output_class'])

        self._check_properties(oracle_data)

        if os.path.exists(self.oracle_data_file_path):
            os.remove(self.oracle_data_file_path)

        oracle_data.to_csv(self.oracle_data_file_path, index=False, header=True)

    def _check_properties(self, data):
        # Implement your property checking logic here
        print("Property checking not implemented. Please add your specific logic here.")
        # Example: Print basic statistics
        print(data.describe())
        print(data[self.params['output_class']].value_counts(normalize=True))


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

adult_data = pd.read_csv(url, names=columns, sep=r'\s*,\s*', engine='python', na_values="?")

categorical_columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                       'native_country', 'income']
numerical_columns = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

protected_attributes = ['age', 'race', 'sex', 'marital_status']

categorical_imputer = SimpleImputer(strategy='most_frequent')
numerical_imputer = SimpleImputer(strategy='median')

adult_data[categorical_columns] = categorical_imputer.fit_transform(adult_data[categorical_columns])
adult_data[numerical_columns] = numerical_imputer.fit_transform(adult_data[numerical_columns])

label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    adult_data[column] = label_encoders[column].fit_transform(adult_data[column])

scaler = StandardScaler()
adult_data[numerical_columns] = scaler.fit_transform(adult_data[numerical_columns])

input_data = pd.DataFrame(adult_data, columns=categorical_columns + numerical_columns)

input_data.rename(columns={'income': 'Class'}, inplace=True)

X = input_data.drop('Class', axis=1)
y = input_data['Class']

iteration_no = 2

for no in range(0, iteration_no):
    model = RandomForestClassifier()
    model.fit(X, y)

    params = {'num_samples': 2000, 'train_ratio': 0.2, 'no_of_params': 2, 'output_class': 'Class',
              'output_col': 'outcome',
              'no_assumption': False, 'no_mapping': False, 'MUTcontent': False}

    params_dict = {
        'max_samples': 1500,
        'deadlines': 500000,
        'white_box_model': 'Decision tree',
        'no_EPOCHS': 20,
        'no_of_params': 2,
        'mul_cex_opt': True,
        'multi_label': False,
        'model_path': '../FairUnAwareTestCases/NBAdult.joblib',
        'model_type': 'sklearn',
        'train_ratio': 30,
        'no_of_train': 1000,
        'train_data_available': True,
        'train_data_loc': '../Datasets/Adult.csv'
    }

    prop_check = PropCheck(model, input_data, params)
    prop_check.run()

    assumptions = []
    statements = []

    assumptions_file_path = '../asm.smt'
    assertions_file_path = '../asrt.smt'

    if os.path.exists(assumptions_file_path):
        os.remove(assumptions_file_path)

    if os.path.exists(assertions_file_path):
        os.remove(assertions_file_path)

    for i, col in enumerate(X.columns.tolist()):
        if col in protected_attributes:
            res_statements = Assume(prop_check.oracle_data_file_path, assumptions_file_path, 'x[i] != y[i]', i)
        else:
            res_statements = Assume(prop_check.oracle_data_file_path, assumptions_file_path, 'x[i] = y[i]', i)

        statements.extend(res_statements)
    Assert(prop_check.oracle_data_file_path, params, prop_check.params, assertions_file_path,
           'model.predict(x) == model.predict(y)')

    obj_faircheck = RunChecker(prop_check.oracle_data_file_path, assumptions_file_path, assertions_file_path, params,
                               params_dict)
    start_time = time.time()
    obj_faircheck.runPropCheck()
    print('time required is', time.time() - start_time)
