import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv('data.csv')
df = df[['text', 'channel', 'date_unixtime']].dropna()

X = df.drop('channel', axis=1)
Y = df.drop(['text', 'date_unixtime'], axis=1)
label_encoder = LabelEncoder()
label_encoder.fit(Y)
Y = label_encoder.transform(Y)
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
label_map = {value : key for key, value in label_map.items()}

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

decision_tree_params = {
  'dt__criterion': ['gini', 'log_loss']
}

trials = [
  ('dt', decision_tree_params, DecisionTreeClassifier())
]

for classifier_name, classifier_params, classifier in trials:
  pipeline = Pipeline(
    [
      ('col_transform', column_transformer),
      ('threshold', VarianceThreshold()),
      ('kbest', SelectKBest()),
      (classifier_name, classifier)
    ]
  )

  param_grid = {
    'kbest__k': [1000],
  } | classifier_params

  grid_search = GridSearchCV(
    pipeline, param_grid, verbose=True, return_train_score=True
  )

  grid_search.fit(X_train, y_train.ravel())
  prediction = grid_search.predict(X_test)

  accuracy = balanced_accuracy_score(y_test, prediction, adjusted=True)
  print('Balanced Accuracy:', accuracy)

  matrix = plot_confusion_matrix(
    grid_search,
    X_test,
    y_test
  )


  '''
  labels = [label_map[i] for i in range(2)]
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')

  matrix.ax_.set_title('Confusion Matrix')
  plt.show()
  plt.clf()
  '''

  plot_tree(grid_search.best_estimator_.steps[-1][1], max_depth = 100)
  plt.show()
