import os
import re
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

csv = "svc_coarse_search.csv"
df = pd.read_csv(
    os.path.join(
        'result',
        csv
    )
)

print(df)

data = df[['param_svc__C', 'param_svc__gamma', 'mean_test_score']]
data_np = data.to_numpy()

fig = plt.figure()
ax = Axes3D(fig)

ax.plot_trisurf(
    data_np[:, 0],
    data_np[:, 1],
    data_np[:, 2],
    cmap=cm.jet
)
plt.savefig(
    os.path.join(
        'result',
        re.sub('csv', 'png', csv)
    )
)
