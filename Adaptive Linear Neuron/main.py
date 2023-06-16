import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from adalinegd import AdalineGD
from adalinesgd import AdalineSGD
import pdr
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

fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(8,4))
ada1=AdalineGD(n_iter=10,eta=0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel('log(Sum-Squared-Error)')
ax[0].set_title("Adaline-learning rate 0.01")
ada2=AdalineGD(n_iter=10,eta=0.0001).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1),ada2.cost_,marker='o')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Sum-Squared-Error")
ax[1].set_title("Adaline - Learning Rate 0.0001")
plt.show()



X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# Create the AdalineGD model
ada = AdalineGD(n_iter = 15, eta = 0.01)

# Train the model
ada.fit(X_std, y)


pdr.plot_decision_regions(X_std, y, classifier = ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.show()

# Plot the training errors of both of the models
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o', color = 'red')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()


ada2 = AdalineSGD(n_iter = 15, eta = 0.01, random_state = 1)

# Train the model
ada2.fit(X_std, y)

# Plot the decision boundary
pdr.plot_decision_regions(X_std, y, classifier = ada2)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.show()

# Plot the training errors of both of the models
plt.plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker = 'x', color = 'blue')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

