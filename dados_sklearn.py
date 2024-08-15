import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import SpectralClustering

seed = 1234

def gera_labels(X, n_clusters=3):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', n_init=10, random_state=seed)
    labels = spectral.fit_predict(X)
    return labels

def gera_subplot(pontos, labels, ax, title):
    scatter = ax.scatter(pontos[:, 0], pontos[:, 1], c=labels, s=10, cmap='viridis')
    ax.set_title(title)

def dados_circulos_1():
    pontos, y = datasets.make_circles(n_samples=700, factor=0.5, noise=0.05, random_state=seed)
    labels = gera_labels(pontos, 2)
    return [2, pontos, labels]

def dados_circulos_2():
    pontos, y = datasets.make_circles(n_samples=1000, factor=0.2, noise=0.1, random_state=seed)
    labels = gera_labels(pontos, 2)
    return [2, pontos, labels]

def dados_moons_1():
    pontos, y = datasets.make_moons(n_samples=700, noise=0.05, random_state=seed)
    labels = gera_labels(pontos, 2)
    return [2, pontos, labels]

def dados_moons_2():
    pontos, y = datasets.make_moons(n_samples=1000, noise=0.1, random_state=seed)
    labels = gera_labels(pontos, 2)
    return [2, pontos, labels]

def dados_blobs_1():
    pontos, y = datasets.make_blobs(n_samples=700, centers=3, random_state=seed)
    return [3, pontos, y]

def dados_blobs_2():
    pontos, y = datasets.make_blobs(n_samples=[150,50,500], centers=None, random_state=seed)
    return [3, pontos, y]

def dados_rng_1():
    rng = np.random.RandomState(seed)
    pontos = rng.rand(700, 2)
    labels = gera_labels(pontos, 4)
    return [4, pontos, labels]

def dados_rng_2():
    rng = np.random.RandomState(seed)
    pontos = rng.rand(1000, 2)
    labels = gera_labels(pontos, 2)
    return [2, pontos, labels]

def dados_anisotropicos_1():
    X, y = datasets.make_blobs(n_samples=500, centers=3, random_state=seed)
    pontos = np.dot(X, [[0.5,-0.5], [-1,2]])
    labels = gera_labels(pontos, 3)
    return [3, pontos, labels]

def dados_anisotropicos_2():
    X, y = datasets.make_blobs(n_samples=500, centers=4, random_state=seed)
    pontos = np.dot(X, [[-0.5,1], [1,2]])
    labels = gera_labels(pontos, 4)
    return [4, pontos, labels]
