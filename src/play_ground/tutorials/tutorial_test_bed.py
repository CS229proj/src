import numpy as  np
from sklearn import datasets
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
  print('hello')
  np.random.seed(0)
  X, y = datasets.make_moons(200, noise=0.20)
  plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

