import os
import re
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

csv = "cnb_coarse_search_small_alpha.csv"
df = pd.read_csv(
    os.path.join(
        'result',
        csv
    )
)


data = df[['param_cnb__alpha', 'param_kbest__k', 'mean_test_score']]
data_np = data.to_numpy()

fig = plt.figure()
ax = Axes3D(fig)

ax.plot_trisurf(
    data_np[:, 0],
    data_np[:, 1],
    data_np[:, 2],
    cmap=cm.jet
)
ax.set_xlabel('alpha')
ax.set_ylabel('k')
ax.set_zlabel('score')

plt.savefig(
    os.path.join(
        'result',
        re.sub('csv', 'png', csv)
    )
)
