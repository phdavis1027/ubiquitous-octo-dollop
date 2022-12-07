import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

features = np.load("features.npy", allow_pickle=True)
print(features.shape)
corr = np.corrcoef(features.astype(float))

sns.heatmap(corr)
plt.show()
