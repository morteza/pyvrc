import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KernelDensity


def plot_kde(kd, x):
  """Utility function to plot KDE scores, supports sklearn and scipy.
  """

  df = pd.DataFrame({
      'x': x.flatten(),
      'score': kd.score_samples(x) if isinstance(kd, KernelDensity) else kd.logpdf(x)
  })

  sns.scatterplot(data=df, x='x', y='score', marker='.')

  plt.ylabel('KDE Score (logpdf)')
  plt.xlabel('X')

  plt.show()
