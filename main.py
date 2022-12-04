total_fits = 0
RESULTS_DIR = './result'

from augment_data import (
  add_gender
)

import datetime
when = datetime.datetime.now()
import os
import time
import json
import re

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation, PCA

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


df = pd.read_csv('data.csv', low_memory=False)
df = df[['text', 'channel', 'date_unixtime']].dropna()

X = df.drop('channel', axis=1)
Y = df.drop(['text', 'date_unixtime'], axis=1)
Y = Y.to_numpy().ravel()
label_encoder = LabelEncoder()
label_encoder.fit(Y)
Y = label_encoder.transform(Y)
label_map_orig = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
label_map = {value : key for key, value in label_map_orig.items()}

X_train, X_test, y_train, y_test = train_test_split(
  X, Y, random_state=1
)

column_transformer = ColumnTransformer(
  [
    ('tfidf', TfidfVectorizer(), 'text')
  ],
  remainder='passthrough'
)

svc_params = {
  'svc__kernel': ['rbf'],
}

trials = [
    ('svc', svc_params, SVC())
]

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
    if not CanaryTransformer.last_time:
      CanaryTransformer.last_time = time.perf_counter()
    self.next_step = next_step

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

color_map = {
  0: 'red',
  1: 'green',
  2: 'blue'
}

X_PCA = column_transformer.fit_transform(
  X
)
X_PCA = PCA(n_components=3).fit_transform(X_PCA.todense())

fig, ax = plt.subplots(projection='3d')

ax.scatter(
  X_PCA[:, 0],
  X_PCA[:, 1],
  X_PCA[:, 2],
  c=[color_map[channel] for channel in Y]
)

plt.figure(fig)
plt.savefig(
  os.path.join(
    RESULTS_DIR,
    re.sub(" ", "", f"{when}-pca.png")
  )
)
print('PCA Projection Plot saved')

def save_params_and_results(
  balanced_accuracy,
  possible_params,
  classifier_name,
  grid_cv
):
  global when
  RESULTS_DIR = './result'
  if not os.path.exists(RESULTS_DIR):
     os.mkdir(RESULTS_DIR)

  with open(
    os.path.join(
      RESULTS_DIR,
      re.sub(' ', '', f'{when}-{classifier_name}.txt')
    ),
    'w+'
  ) as f:
    f.write(f"[BALANCED ACCURACY] - [{balanced_accuracy}]\n")
    f.write(f"[PIPELINE STEPS]\n")
    f.write(f"{grid_cv.best_estimator_.steps}")

  pd.DataFrame(grid_cv.cv_results_).to_csv(os.path.join(RESULTS_DIR, re.sub(' ', '', f'{when}-results.cv')))

  ### Save a projection of the whole dataset
for classifier_name, classifier_params, classifier in trials:
  pipeline = Pipeline(
    [
      ('can_0', CanaryTransformer("Column Transformer (TFIDF on column 'text')")),
      ('col_transform', column_transformer),
      ('can_1', CanaryTransformer("StandardScaler")),
      ('std_scaler', StandardScaler(with_mean=False)),
      ('can_2', CanaryTransformer("VarianceThreshold")),
      ('threshold', VarianceThreshold()),
      ('can_3', CanaryTransformer('SelectKBest')),
      ('kbest', SelectKBest()),
      ('can_4', CanaryTransformer(classifier_name)),
      (classifier_name, classifier)
    ]
  )
  print('Exiting pipeline...')

  param_grid = {
    'kbest__k': [600, 700, 1000]
  } | classifier_params

  grid_search = GridSearchCV(
    pipeline, param_grid, verbose=True, return_train_score=True, scoring='balanced_accuracy'
  )

  grid_search.fit(X_train, y_train.ravel())
  prediction = grid_search.predict(X_test)

  accuracy = balanced_accuracy_score(y_test, prediction, adjusted=True)
  print("[Balanced Accuracy]", accuracy)

  save_params_and_results(
    accuracy,
    param_grid,
    classifier_name,
    grid_search
  )

  matrix = ConfusionMatrixDisplay.from_estimator(
    grid_search,
    X_test,
    y_test
  )

  plt.figure(matrix.figure_)

  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')

  matrix.ax_.set_title('Confusion Matrix')
  plt.savefig(
    os.path.join(
      RESULTS_DIR,
      re.sub(" ", "", f"{when}-confusion.png")
    )
  )
  print(f'Confusion Matrix for {classifier_name} saved.')

