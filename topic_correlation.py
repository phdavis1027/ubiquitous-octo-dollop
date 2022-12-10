import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold

from custom_transformers import LdaTransformer, NmfTransformer, TSvdTransformer

df = pd.read_csv('data.csv', low_memory=False)
df = df[['text', 'channel', 'date_unixtime']].dropna()
df = df.drop('date_unixtime', axis=1)
df['channel'] = df['channel'].to_numpy().ravel()
label_encoder = LabelEncoder()
label_encoder.fit(df['channel'])
df['channel'] = label_encoder.transform(df['channel'])
label_map_orig = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
label_map = {value : key for key, value in label_map_orig.items()}

col_transformer_lda = ColumnTransformer(
    [
        ('lda', LdaTransformer(), 'text')
    ],
    remainder='passthrough'
)

col_transformer_nmf = ColumnTransformer(
    [
        ('nmf', NmfTransformer(), 'text')
    ],
    remainder='passthrough'
)

col_transformer_tsvd = ColumnTransformer(
    [
        ('tsvd', TSvdTransformer(), 'text')
    ],
    remainder='passthrough'
)

topic_analyzers = [
    ('LDA', col_transformer_lda),
    ('NMF', col_transformer_nmf),
    ('TSVD', col_transformer_tsvd)
]

fig, axs = plt.subplots(1, len(topic_analyzers))

for i, (name, analyzer) in enumerate(topic_analyzers):
    model = Pipeline(
        [
            ('col_transform', analyzer),
            ('std_scaler', StandardScaler(with_mean=False)),
            ('threshold', VarianceThreshold())
        ]
    )

    X = model.fit_transform(df)
    # print(X)
    corr = np.corrcoef(X)
    print(corr.shape)
    print(corr)

    sns.heatmap(data=corr[3000:6000, 3000:6000], ax=axs[i])
    axs[i].set_title(f'Correlation of topics - {name}')

plt.show()
