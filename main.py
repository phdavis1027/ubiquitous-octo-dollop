total_fits = 0
RESULTS_DIR = './result'

import seaborn as sns

from custom_transformers import *

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
from sklearn.decomposition import LatentDirichletAllocation, PCA

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import ComplementNB


df = pd.read_csv('data.csv', low_memory=False)
# df = clean_text(df)
# df = add_msg_sentiment(df)
# df = add_msg_length(df)
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

'''
  'svc__C': [2**k for k in np.arange(4, 6, 0.1)],
  'svc__gamma': [2**k for k in np.arange(-15, -12, 0.25)]
'''

svc_params = {
  'svc__kernel': ['rbf'],
  'svc__C':[59.7141114583554],
  'svc__gamma': [0.000205296976380301]
}

cnb_params = {
  'cnb__alpha': [.03],
}

trials = [
  ('svc', svc_params, SVC())
]

column_transformer_topify = ColumnTransformer(
  [
    ('topify', LdaTransformer(), 'text')
  ],
  remainder='passthrough'
)

column_transformer = ColumnTransformer(
  [
    ('tfidf', TfidfVectorizer(), 'text')
  ],
  remainder='passthrough'
)


if len(sys.argv) > 1 and sys.argv[1] == 'pca':
  color_map = {
    0: 'red',
    1: 'green',
    2: 'blue'
  }

  X_PCA = column_transformer.fit_transform(
    X
  )
  X_PCA = PCA(n_components=3).fit_transform(X_PCA.todense())

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(
    X_PCA[:, 0],
    X_PCA[:, 1],
    X_PCA[:, 2],
    c=[color_map[channel] for channel in Y]
  )
  ax.legend(
    [label_map[i] for i in range(2)],
    [color_map[i] for i in range(2)]
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
      'svc_coarse_search_ngram_range.txt'
    ),
    'w+'
  ) as f:
    f.write(f"[BALANCED ACCURACY] - [{balanced_accuracy}]\n")
    f.write(f"[PIPELINE STEPS]\n")
    f.write(f"{grid_cv.best_estimator_.steps}")

  pd.DataFrame(grid_cv.cv_results_).to_csv("result/svc_coarse_search_ngram_range.csv")

  ### Save a projection of the whole dataset
for classifier_name, classifier_params, classifier in trials:
  n_jobs = multiprocessing.cpu_count() - 1
  pipeline = Pipeline(
    [
      ('col_transform', column_transformer),
      ('std_scaler', StandardScaler(with_mean=False)),
      ('threshold', VarianceThreshold()),
      ('kbest', SelectKBest()),
      (classifier_name, classifier)
    ]
  )

  if n_jobs > 1:
    pipeline.steps = list(filter(lambda trans: not isinstance(trans[1], CanaryTransformer), pipeline.steps))
  print([(i, j) for i in range(4) for j in range (i, 4)])

  param_grid = {
    'kbest__k': [1000],
    'col_transform__tfidf__ngram_range': [(i, j) for i in range(1, 15) for j in range (i, 15)]
  } | classifier_params

  grid_search = GridSearchCV(
    pipeline,
    param_grid,
    verbose=True,
    return_train_score=True,
    scoring='balanced_accuracy',
    n_jobs=n_jobs
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
      re.sub(" ", "", f"svc_coarse_search_ngram_range.png")
    )
  )
  print(f'Confusion Matrix for {classifier_name} saved.')

