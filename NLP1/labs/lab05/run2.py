from sklearn.neural_network import MLPClassifier
model = MLPClassifier()

for dataset in datasets:
    X, y = dataset["data"]
    model.fit(X, y)
    print("Accuracy on the dataset ’%s’ was %.1f%%" %
            (dataset["title"], 100*model.score(X, y)))
