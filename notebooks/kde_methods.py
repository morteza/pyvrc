"""Explode different methods to calculate stable KDE of a given dataset."""

# %%

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
import seaborn as sns


# X = (n_samples * n_features)
X = np.random.randn(1000).reshape(-1, 1) * 10
n = X.shape[1]
d = X.shape[0]
bw = (n * (d + 2) / 4.)**(-1. / (d + 4))  # silverman
# bw = n**(-1. / (d + 4))  # scott


kde_scipy = stats.gaussian_kde(X.T, bw_method='silverman')
kde_sklearn = KernelDensity(kernel='gaussian', bandwidth=bw)

kde_sklearn.fit(X)

df = pd.DataFrame({
    'X': X.flatten(),
    'scipy': kde_scipy.logpdf(X.T),
    'sklearn': kde_sklearn.score_samples(X)
})

df = df.melt('X', var_name='method', value_name='score')

sns.scatterplot(data=df, x='X', y='score', hue='method', marker='.')

plt.ylabel('KDE Score (logpdf)')
plt.xlabel('X')

plt.show()

# benchmarks (uncomment to run benchmarks)

# print('scipy kde benckmark:')
# %%timeit kd = stats.gaussian_kde(X.T, bw_method='scott'); kd.logpdf(X.T)

# print('sklearn kde benckmark:')
# %%timeit kd = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X); kd.score_samples(X)
