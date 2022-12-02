import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import LatentDirichletAllocation, PCA

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv('data.csv')
data = df[['channel', 'date_unixtime']].dropna().to_numpy()

label = {
  'tka':0,
  'psf':1,
  'gc':2
}

ax = plt.gcf().add_subplot()
ax.scatter(
  np.array([label[ch] for ch in data[:, 0]]),
  data[:, 1],
  c=np.array([label[ch] for ch in data[:, 0]]),
)

plt.show()
