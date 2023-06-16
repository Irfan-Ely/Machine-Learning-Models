import numpy as np

class Perceptron(object):
   def __init__(self, rate = 0.01, niter = 10):
      self.rate = rate
      self.niter = niter

   def fit(self, X, y):
      # weights
      self.weight = np.zeros(1 + X.shape[1])
      # Number of misclassifications
      self.errors = []  # Number of misclassifications

      for i in range(self.niter):
         err = 0
         for xi, target in zip(X, y):
            delta_w = self.rate * (target - self.predict(xi))
            self.weight[1:] += delta_w * xi
            self.weight[0] += delta_w
            err += int(delta_w != 0.0)
         self.errors.append(err)
      return self

   def net_input(self, X):
      """Calculate net input"""
      # print("X:",X)
      # print("Weight:",self.weight[1:])
      # print("Dot Product:", np.dot(X, self.weight[1:]))
      # print("Weight[0]:", self.weight[0])
      # print("Netinput:",np.dot(X, self.weight[1:]) + self.weight[0])
      return np.dot(X, self.weight[1:]) + self.weight[0]


   def predict(self, X):
      """Return class label after unit step"""
      return np.where(self.net_input(X) >= 0.0, 1, -1)

import pandas as pd
df=pd.read_csv("./iris.data",header=None)
print(df.tail())
#Section # 02
import matplotlib.pyplot as plt
import numpy as np
y=df.iloc[0:100,4].values

#
y=np.where(y=="Iris-setosa",-1,1)

X=df.iloc[0:100,[0,2]].values
print(np.zeros(1+X.shape[1]))
plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label="setosa")
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label="versicolor")
plt.xlabel("Petal Length")
plt.ylabel("Sepal Length")
plt.legend(loc="upper left")
plt.show()
# #
# #Section # 03
pn = Perceptron(0.1, 10)
pn.fit(X, y)
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
#
#
#
# #Section # 04
# from matplotlib.colors import ListedColormap
# def plot_decision_regions(X, y, classifier, resolution=0.02):
#    # setup marker generator and color map
#    markers = ('s', 'x', 'o', '^', 'v')
#    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#    cmap = ListedColormap(colors[:len(np.unique(y))])
#
#    # plot the decision surface
#    x1_min, x1_max = X[:,  0].min() - 1, X[:, 0].max() + 1
#    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#    np.arange(x2_min, x2_max, resolution))
#    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#    plt.xlim(xx1.min(), xx1.max())
#    plt.ylim(xx2.min(), xx2.max())
#
#    # plot class samples
#    for idx, cl in enumerate(np.unique(y)):
#       plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
#       alpha=0.8, c=cmap(idx),
#       marker=markers[idx], label=cl)
#
#
#
# plot_decision_regions(X, y, classifier=pn)
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left')
# plt.show()