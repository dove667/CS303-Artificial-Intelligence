import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import os
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:

    def __init__(self, data_dir='.'):

        self.data_dir = data_dir
        self.categorical_features = [
            'workclass', 'education', 'marital.status', 'occupation', 
            'relationship', 'race', 'sex', 'native.country'
        ]
        self.numerical_features = [
            'age', 'fnlwgt', 'education.num', 'capital.gain', 
            'capital.loss', 'hours.per.week'
        ]
        self.cat_imputer = None
        self.num_imputer = None
        self.scaler = None
        self.encoder = None
        self.feature_columns = None
    
    def load_data(self, ):

        print("=" * 60)
        print("Loading data...")
        print("=" * 60)
        
        train_data = pd.read_csv(os.path.join(self.data_dir, 'traindata.csv'))
        test_data = pd.read_csv(os.path.join(self.data_dir, 'testdata.csv'))
        with open(os.path.join(self.data_dir, 'trainlabel.txt'), 'r') as f:
            labels = [int(label.strip()) for label in f.readlines()]
        
        print(f"\nTraining data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")
        
        print(f"\nTotal positive labels: {sum(labels)}")
        print(f"Total negative labels: {len(labels) - sum(labels)}")
        print(f"Positive class ratio: {sum(labels) / len(labels):.4f}")
        
        return train_data, test_data, labels
    
    def analyze_missing_values(self, train_data, test_data):

        print("\n" + "=" * 60)
        print("Analyzing missing values (represented as '?')...")
        print("=" * 60)
        
        def count_missing(df, features):
            missing = {}
            for feature in features:
                if feature in df.columns:
                    count = (df[feature] == '?').sum() if df[feature].dtype == 'object' else df[feature].isna().sum()
                    if count > 0:
                        missing[feature] = count
            return missing
        
        train_missing_cat = count_missing(train_data, self.categorical_features)
        train_missing_num = count_missing(train_data, self.numerical_features)
        test_missing_cat = count_missing(test_data, self.categorical_features)
        test_missing_num = count_missing(test_data, self.numerical_features)
        
        if train_missing_cat:
            print(f"Training categorical features with missing values: {train_missing_cat}")
        if train_missing_num:
            print(f"Training numerical features with missing values: {train_missing_num}")
        if test_missing_cat:
            print(f"Testing categorical features with missing values: {test_missing_cat}")
        if test_missing_num:
            print(f"Testing numerical features with missing values: {test_missing_num}")
    
    def handle_missing_values(self, train_data, test_data, fit=True):

        print("\n" + "=" * 60)
        print("Handling missing values with mode imputation...")
        print("=" * 60)
        
        if fit:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            train_data[self.categorical_features] = self.cat_imputer.fit_transform(
                train_data[self.categorical_features]
            )
        else:
            train_data[self.categorical_features] = self.cat_imputer.transform(
                train_data[self.categorical_features]
            )
        
        test_data[self.categorical_features] = self.cat_imputer.transform(
            test_data[self.categorical_features]
        )
        
        if fit:
            self.num_imputer = SimpleImputer(strategy='most_frequent')
            train_data[self.numerical_features] = self.num_imputer.fit_transform(
                train_data[self.numerical_features]
            )
        else:
            train_data[self.numerical_features] = self.num_imputer.transform(
                train_data[self.numerical_features]
            )
        
        test_data[self.numerical_features] = self.num_imputer.transform(
            test_data[self.numerical_features]
        )
        
        print("Missing values handled successfully")
        return train_data, test_data
    
    def feature_engineering(self, train_data, test_data):

        print("\n" + "=" * 60)
        print("Performing feature engineering...")
        print("=" * 60)
        
        train_data['age_group'] = pd.cut(
            train_data['age'], 
            bins=[0, 18, 35, 50, 65, 100], 
            labels=['lt18', '18-35', '35-50', '50-65', 'gt65']
        ).astype(str)
        test_data['age_group'] = pd.cut(
            test_data['age'],
            bins=[0, 18, 35, 50, 65, 100],
            labels=['lt18', '18-35', '35-50', '50-65', 'gt65']
        ).astype(str)
        print("Created feature: age_group")
        
        train_data['capital_total'] = train_data['capital.gain'] + train_data['capital.loss']
        test_data['capital_total'] = test_data['capital.gain'] + test_data['capital.loss']
        print("Created feature: capital_total")
        
        train_data['has_capital'] = (train_data['capital_total'] > 0).astype(int)
        test_data['has_capital'] = (test_data['capital_total'] > 0).astype(int)
        print("Created feature: has_capital")
        
        return train_data, test_data
    
    def remove_low_predictive_features(self, train_data, test_data, features_to_remove=['fnlwgt']):

        print("\n" + "=" * 60)
        print("Removing features with low predictive power...")
        print("=" * 60)
        
        for feature in features_to_remove:
            if feature in train_data.columns:
                train_data = train_data.drop(feature, axis=1)
                if feature in test_data.columns:
                    test_data = test_data.drop(feature, axis=1)
                print(f"Removed feature: {feature}")
        
        return train_data, test_data
    
    def encode_categorical_features(self, train_data, test_data, fit=True):

        print("\n" + "=" * 60)
        print("Encoding categorical features (One-Hot)...")
        print("=" * 60)
        
        categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
        print(f"Detected categorical columns: {categorical_cols}")
        
        if fit:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            train_encoded_arr = self.encoder.fit_transform(train_data[categorical_cols])
        else:
            train_encoded_arr = self.encoder.transform(train_data[categorical_cols])
        
        encoded_cols = self.encoder.get_feature_names_out(categorical_cols)
        
        train_encoded = pd.DataFrame(train_encoded_arr, columns=encoded_cols, index=train_data.index)
        
        non_categorical_cols = [col for col in train_data.columns if col not in categorical_cols]
        for col in non_categorical_cols:
            train_encoded[col] = train_data[col].values
        
        test_encoded_arr = self.encoder.transform(test_data[categorical_cols])
        test_encoded = pd.DataFrame(test_encoded_arr, columns=encoded_cols, index=test_data.index)
        
        for col in non_categorical_cols:
            test_encoded[col] = test_data[col].values
        
        print(f"Encoded to {len(train_encoded.columns)} features")
        
        return train_encoded, test_encoded
    
    def scale_numerical_features(self, train_data, test_data, fit=True):

        print("\n" + "=" * 60)
        print("Scaling numerical features...")
        print("=" * 60)
        
        numerical_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            self.scaler = StandardScaler()
            train_data[numerical_cols] = self.scaler.fit_transform(train_data[numerical_cols])
        else:
            train_data[numerical_cols] = self.scaler.transform(train_data[numerical_cols])
        
        test_data[numerical_cols] = self.scaler.transform(test_data[numerical_cols])
        
        print(f"Scaled {len(numerical_cols)} numerical features")
        
        return train_data, test_data
    
    def preprocess(self, feature_engineering=False, remove_low_predictive=False, fit=True):

        train_data, test_data, y_train = self.load_data()
        
        self.analyze_missing_values(train_data, test_data)
        
        train_data, test_data = self.handle_missing_values(train_data, test_data, fit=fit)

        if feature_engineering:
            train_data, test_data = self.feature_engineering(train_data, test_data)
        
        if remove_low_predictive:
            train_data, test_data = self.remove_low_predictive_features(train_data, test_data)
        
        train_data, test_data = self.encode_categorical_features(train_data, test_data, fit=fit)
        
        X_train, X_test = self.scale_numerical_features(train_data, test_data, fit=fit)
        
        self.feature_columns = X_train.columns.tolist()
        
        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)
        
        return X_train, y_train, X_test

