import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn import preprocessing
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from os.path import exists

columns = ['age', 'workclass', 'fnlwht', 'education', 'education-num', 'marital-status', 'occupation',
          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
          'income']

def read_data(filepath):
    data = pd.read_csv(
        filepath,
        names=columns,
        sep=r'\s*,\s*',
        engine='python', skiprows=1,
        na_values="?")
    return data

def remove_fullstop(s):
  if s[-1] == '.':
    s = s[:-1]
  return s

def preprocess(full_data):
    # Transform labels
    full_data['income'] = np.where(full_data["income"].str.contains(">50K"), 1, 0)
    
    # Delete rows where occupation is null.
    full_data = full_data[full_data['occupation'].notnull()]

    # Fill rows with missing country with typical value (United States).
    full_data['native-country'] = full_data['native-country'].fillna('United-States')

    # Check that no rows have null values.
    print(full_data.isna().sum())

    # Drop fnlwhat column.
    full_data.drop(columns=['fnlwht'], inplace=True)

    # Segment out target label
    full_labels = full_data['income']
    full_data.drop(columns=['income'], inplace=True)

    # Deal with categorical variables
    # Segment categorical and non categorical data (will manipulate cat_data, and append them back later)
    cat_data = full_data.select_dtypes(include=['object']).copy()
    other_data = full_data.select_dtypes(include=['int']).copy()

    # One hot other categorical variables
    newcat_data = pd.get_dummies(cat_data, columns=[
        "workclass", "education", "native-country" ,"relationship", "marital-status", "occupation", "race", "sex"
    ])

    print(newcat_data.head())

    # Append all columns back together
    full_data = pd.concat([other_data, newcat_data], axis=1)
    print(full_data.head())

    # Remove basis columns to remove linear dependence
    col_del = ['workclass_Private', 'education_HS-grad', 'marital-status_Married-civ-spouse', 
            'occupation_Prof-specialty', 'relationship_Husband', 'race_White']

    df_data = full_data.drop(columns=col_del, axis=1)
    df_labels = full_labels

    return df_data, df_labels