# 
Model 1: Learning from disaster - Build a machine learning algorithm to predict which passengers survived the shipwreck.

Model 2: perform a classification of mails as either spam or not.
##Libraries used

#linear algebra

import numpy as np

#data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 

#Visualization libraries

import seaborn as sns

sns.set_style('darkgrid')

import matplotlib.pyplot as plt

import missingno as mno

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

from sklearn.neighbors import KNeighborsClassifier ## KNN classifier algorithm

from sklearn.model_selection import train_test_split ## Train and test split algorithm

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score # Model performance 
metrics

from sklearn.naive_bayes import GaussianNB # Naive Bayes algorithm

from sklearn.preprocessing import Normalizer # For normalizing data to unit variance

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # machine learning algorithm that can also be used 
as a feature reduction preprocessing tool.
