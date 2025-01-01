import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

class HyperspectralImageClusterer:
    def __init__(self, data_dir, tif_filename, output_filename,shapefile_path, n_clusters=12, n_components=20):
        self.data_dir = data_dir
        self.tif_file = os.path.join(data_dir, tif_filename)
        self.output_path = os.path.join(data_dir, output_filename)
        self.shapefile_path = os.path.join(data_dir, shapefile_path)
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.hyperspectral_image = None
        self.profile = None
        self.cluster_labels = None

    def load_image(self):
        with rasterio.open(self.tif_file) as src:
            self.hyperspectral_image = src.read()
            self.profile = src.profile
        print("Image loaded.")

    def perform_mnf(self):
        n_bands, height, width = self.hyperspectral_image.shape
        X = self.hyperspectral_image.reshape(n_bands, -1).T

        # Estimate noise covariance
        noise_cov = np.cov(np.diff(X, axis=0).T)
        noise_cov += np.eye(noise_cov.shape[0]) * 1e-10

        # Whiten the data
        eig_values, eig_vectors = np.linalg.eigh(noise_cov)
        whitening = eig_vectors.dot(np.diag(1.0 / np.sqrt(eig_values))).dot(eig_vectors.T)
        X_whitened = X.dot(whitening)

        

        # Perform PCA on whitened data        
        pca = PCA(n_components=self.n_components)
        X_mnf = pca.fit_transform(X_whitened)

        print("MNF transformation performed.")
        return X_mnf

    def perform_pca(self):
        n_bands, height, width = self.hyperspectral_image.shape
        X = self.hyperspectral_image.reshape(n_bands, -1).T

        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(X)

        print("PCA transformation performed.")
        return X_pca

    def perform_clustering(self, X_reduced):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(X_reduced)

        height, width = self.hyperspectral_image.shape[1:]
        self.cluster_labels = kmeans.labels_.reshape(height, width)
        print("Clustering performed.")

    def save_clustered_image(self):
        self.profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

        with rasterio.open(self.output_path, 'w', **self.profile) as dst:
            dst.write(self.cluster_labels.astype(rasterio.uint8), 1)
        print("Clustered image saved.")

    def convert_to_shapefile(self):
        # Read the clustered image
        with rasterio.open(self.output_path) as src:
            image = src.read(1)
            transform = src.transform

        # Generate shapes
        shapes = rasterio.features.shapes(image, transform=transform)

        # Convert shapes to GeoDataFrame
        geoms = []
        for geom, value in shapes:
            if value > 0:  # Filter out background
                geoms.append({"geometry": shape(geom), "value": value})

        gdf = gpd.GeoDataFrame(geoms, crs=self.profile['crs'])

        # Save to shapefile
        gdf.to_file(self.shapefile_path)
        print("Shapefile saved.")

    def process(self, method='mnf'):
        self.load_image()
        if method == 'mnf':
            X_reduced = self.perform_mnf()
        elif method == 'pca':
            X_reduced = self.perform_pca()
        else:
            raise ValueError("Method must be either 'mnf' or 'pca'")
        self.perform_clustering(X_reduced)
        self.save_clustered_image()
        self.convert_to_shapefile()

if __name__ == "__main__":
    data_dir = r'D:\work\PRISMA Group\00 project\data\Seenaa02\PRS_L2D_STD_20201020083329_20201020083333_0001'
    tif_filename = 'PRS_L2D_STD_20201020083329_20201020083333_0001.tif'
    output_filename = 'PRS_L2D_STD_20201020083329_20201020083333_0001_clustered_image_mnf.tif'
    shapefile_path  = 'PRS_L2D_STD_20201020083329_20201020083333_0001_clustered.shp'

    clusterer = HyperspectralImageClusterer(data_dir, tif_filename, output_filename,shapefile_path, n_components=20)
    
    # Use MNF
    clusterer.process(method='mnf')
    
    # Alternatively, use PCA
    # clusterer.process(method='pca')