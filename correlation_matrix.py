import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

features = np.load("features.npy", allow_pickle=True)
print(features)
corr = np.corrcoef(features.astype(float))

sns.heatmap(corr)

plt.savefig('result/cnb_correlation_1000_by_1000.png')
