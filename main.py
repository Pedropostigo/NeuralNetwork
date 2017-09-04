import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deepnet import NNetwork

data = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

labels = data['label']
data = data.drop('label', 1)


def plotnumber(number):

    x = int(np.sqrt(len(number)))
    matrix = np.asarray(number.values.reshape(x, x))

    plt.imshow(matrix, cmap = "gray")
    plt.show()

net = NNetwork([500, 500, 500])
net.train(data, labels)
pred = net.predict(test)

prediction = pd.DataFrame({'ImageId': [i + 1 for i in range(0, test.shape[0])],
                           'Label': pred})

prediction.to_csv("data/output.csv", index = False)

for i in range(20):
    print(pred[i])
    plotnumber(test.loc[i, :])
