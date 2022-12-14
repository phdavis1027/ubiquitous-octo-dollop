import seaborn as sns

import threading
import multiprocessing
import sys
import datetime
when = datetime.datetime.now()
import os
import time
import json
import re

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA, TruncatedSVD

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import ComplementNB

class TSvdTransformer(TransformerMixin):
  def __init__(self, n_components=10):
    self.n_components = n_components
    self.params = {
      'n_components': n_components
    }

  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, X, y=None, **fit_params):
    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(X)
    np.save('corr_tfidf.npy', vec.get_feature_names_out())
    svd = TruncatedSVD(n_components=self.n_components).fit(X)
    self.components_ = svd.components_
    X = svd.transform(X)
    self.features = X

    return X

  def get_params(self, deep=False):
    return self.params

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self

class NmfTransformer(TransformerMixin):
  def __init__(self, n_components=10):
    self.n_components = n_components
    self.params = {
      'n_components': n_components
    }

  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, X, y=None, **fit_params):
    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(X)
    np.save('corr_tfidf.npy', vec.get_feature_names_out())
    nmf = NMF(n_components=self.n_components).fit(X)
    self.components_ = nmf.components_
    X = nmf.transform(X)
    self.features = X
    return X

  def get_params(self, deep=False):
    return self.params

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self

class LdaTransformer(TransformerMixin):
  def __init__(self, n_components=10):
    self.n_components = n_components
    self.params = {
      'n_components': n_components
    }

  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, X, y=None, **fit_params):
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(X)
    np.save('corr_count.npy', vec.get_feature_names_out())
    lda = LatentDirichletAllocation(n_components=self.n_components).fit(X)
    self.components_ = lda.components_
    X = lda.transform(X)
    self.features = X
    print('lda shape')
    print(X.shape)

    return X

  def get_params(self, deep=False):
    return self.params

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self

class NDArraySaver(TransformerMixin):
  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, X, y=None, **fit_params):
    np.save('features.npy', X)
    return X

class DenseTransformer(TransformerMixin):
  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, X, y=None, **fit_params):
    print('Densifying')
    return X.todense()

class SparseTransformer(TransformerMixin):
  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, X, y=None, **fit_params):
    print('Sparsifying')
    return sparse.csr_matrix(X)

canary_num = 0
class CanaryTransformer(TransformerMixin):
  last_time = None
  def __init__(self, next_step, by=1, columns=None):
    self.next_step = next_step
    if not CanaryTransformer.last_time:
      CanaryTransformer.last_time = time.perf_counter()

  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, X, y=None, **fit_params):
    global canary_num
    global total_fits
    print('[CANARY] About to launch', self.next_step, '...')
    print('This is iteration', canary_num, 'of', total_fits)
    canary_num += 1

    cur_time = time.perf_counter()
    elapsed_time = cur_time - CanaryTransformer.last_time
    print('Last step took about', elapsed_time, 'seconds')
    print('---')
    CanaryTransformer.last_time = cur_time

    return X

