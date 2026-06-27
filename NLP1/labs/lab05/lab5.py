#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fredrik Wahlberg <fredrik.wahlberg@lingfil.uu.se>
"""
import numpy as np
import matplotlib.pyplot as plt


#%% Create the datasets
def make_datasets():
    from sklearn.datasets import make_circles, make_classification, make_moons
    
    n_samples = 200
    data = list()
    data.append({'title': "2 classes", 
                 'data': make_classification(n_samples=n_samples, n_classes=2, 
                                             n_features=2, n_redundant=0, n_informative=2, 
                                             random_state=3, n_clusters_per_class=1)})
    data.append({'title': "3 classes", 
                 'data': make_classification(n_samples=n_samples, n_classes=3, 
                                             n_features=2, n_redundant=0, n_informative=2, 
                                             random_state=3, n_clusters_per_class=1)})
    data.append({'title': "Moons", 
                 'data': make_moons(n_samples=n_samples, noise=0.1, random_state=0)})
    data.append({'title': "Circles", 
                 'data': make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=0)})
    theta = np.linspace(np.pi/2, np.pi*3/2, num=n_samples//2)
    X = np.concatenate([np.zeros((n_samples//2, 2)), np.column_stack([np.cos(theta), np.sin(theta)])])
    y = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    X += np.random.normal(scale=0.1, size=X.shape)
    data.append({'title': "Moon and blob", 
                 'data': (X, y)})
    X = np.random.uniform(-1, 1, size=(n_samples, 2))
    y = np.asarray([1 if (X[i, 0]>0 or X[i, 1]>0) and not (X[i, 0]>0 and X[i, 1]>0) else 0 for i in range(n_samples)])
    data.append({'title': "XOR", 
                 'data': (X, y)})
    return data


def plot_classification(dataset, classifier=None, ax=None):
    """Plots a dataset as a scatterplot
    
    classifier: Points as a contour plot of predictions.
    ax: Axis for plotting, creates a new axis if none is provided."""
    if ax is None:
        fig, axis = plt.subplots(1, 1)
    else:
        axis = ax
    markers = ['o', 's', 'd']
    X, y = dataset['data']
    for j, label in enumerate(set(y)):
            axis.scatter(X[y==label, 0], X[y==label, 1], c=y[y==label], s=50, 
                       marker=markers[j], edgecolor='k', cmap='coolwarm', 
                       vmin=y.min(), vmax=y.max())
    x_min, x_max, y_min, y_max = axis.axis('equal')
    if classifier is not None:
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=200), np.linspace(y_min, y_max, num=200))
        Z = classifier.predict(np.column_stack([xx.ravel(), yy.ravel()]))
        # if hasattr(classifier, "decision_function"):
        #     Z = classifier.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
        # else:
        #     Z = classifier.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]
        Z = Z.reshape(xx.shape)
        axis.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.8, zorder=-1)
        # TODO Fix 3 class contour
    axis.set_title(dataset['title'])
    if ax is None:
        plt.show()


