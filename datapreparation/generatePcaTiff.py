import numpy as np
import rasterio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

data_dir = r'F:\Senaa\Seenaa02\PRS_L2D_STD_20201020083329_20201020083333_0001'
tif_file = os.path.join(data_dir, 'PRS_L2D_STD_20201020083329_20201020083333_0001.tif')
pcatif_file = os.path.join(data_dir, 'PRS_L2D_STD_20201020083329_20201020083333_0001_pca3.tif')

# Load hyperspectral image
with rasterio.open(tif_file) as src:
    hyperspectral_data = src.read()
    profile = src.profile

# Reshape the data to (pixels, bands)
bands, height, width = hyperspectral_data.shape
reshaped_data = hyperspectral_data.reshape(bands, height * width).T

# Apply PCA
n_components = 3
pca = PCA(n_components)
principal_components = pca.fit_transform(reshaped_data)

# Plot explained variance ratio
plt.figure(figsize=(8, 5))
plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7, align='center')
plt.title('Explained Variance Ratio by PCA Components')
plt.xlabel('PCA Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, n_components + 1))
plt.show()

# Normalize the principal components to be greater than zero
min_vals = principal_components.min(axis=0)
principal_components -= min_vals

# Scale to uint16 range
max_val = principal_components.max()
pca_image_scaled = ((principal_components / max_val) * 65535).astype(np.uint16)

# Reshape the principal components back to image format
pca_image = pca_image_scaled.T.reshape(n_components, height, width)

# Prepare the profile for the output TIFF
profile.update(count=n_components, dtype='uint16')

# Save the PCA components as a TIFF image
with rasterio.open(pcatif_file, 'w', **profile) as dst:
    dst.write(pca_image)

# Optionally, display the first PCA component
plt.imshow(pca_image[0], cmap='gray')
plt.title('First PCA Component')
plt.show()