import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.axes_grid1 import ImageGrid


import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg' depending on your system

# Helper function to plot images at scatter plot positions
def plot_images_on_scatter(X, images, ax, zoom=0.5):
    for i in range(len(images)):
        imagebox = OffsetImage(images[i], zoom=zoom)
        ab = AnnotationBbox(imagebox, (X[i, 0], X[i, 1]), frameon=False)
        ax.add_artist(ab)

# Load your images here (for simplicity, generating random images)
n_images = 100
image_size = (28, 28)
images = np.random.rand(n_images, *image_size)

# Flatten the images for dimensionality reduction
X = images.reshape(n_images, -1)

# Standardize the features
X_scaled = StandardScaler().fit_transform(X)

# Dimensionality reduction using PCA or t-SNE
# For PCA:
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# For t-SNE (uncomment to use):
# tsne = TSNE(n_components=2, random_state=42)
# X_reduced = tsne.fit_transform(X_scaled)

# Create a scatter plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(X_reduced[:, 0], X_reduced[:, 1])

# Plot a subset of images on the scatter plot
plot_images_on_scatter(X_reduced, images, ax, zoom=0.5)

plt.show()