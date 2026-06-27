from lab5 import make_datasets
from lab5 import plot_classification
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

model = MLPClassifier(hidden_layer_sizes = (2,3,4), alpha = np.logspace(-2,-6,10))
datasets = make_datasets()

title = datasets[0]["title"]
X, y = datasets[0]["data"]

#fig, ax = plt.subplots(1, 1)
plot_classification(datasets[0])
#plt.show()

for dataset in datasets:
    X, y = dataset["data"]
    model.fit(X, y)
    print("Accuracy on the dataset ’%s’ was %.1f%%" %
            (dataset["title"], 100*model.score(X, y)))


