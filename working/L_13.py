import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.manifold import Isomap


from sklearn.manifold import TSNE


digits = datasets.load_digits()
print(digits.images.shape)
print(digits.keys())

print(digits.target[100])
print(digits.images[100])


model = Isomap(n_neighbors=7, n_components=5)

X = digits.data
print(X.shape)

X_reduced = model.fit_transform(X)
print("Reduced dataset shape:", X_reduced.shape)

# Scatter plot di X_reduced con colormap basata sul target
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=digits.target, cmap='tab10', s=30, alpha=0.8)
plt.colorbar(scatter, label='Target')
plt.title('Isomap Projection of Digits Dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()
plt.close()

model = Isomap(n_neighbors=7, n_components=5)
X_reduced = model.fit_transform(X)

tsne_model = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne_model.fit_transform(X_reduced)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=digits.target, cmap='tab10', s=30, alpha=0.8)
plt.colorbar(scatter, label='Target')
plt.title('t-SNE Projection of Isomap Reduced Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()