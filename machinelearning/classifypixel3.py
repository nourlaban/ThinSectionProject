import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, label, generate_binary_structure
from sklearn.decomposition import PCA

class MaskProcessor:
    def __init__(self, mask, min_size=64, hole_size=64, sigma=1):
        self.mask = mask
        self.min_size = min_size
        self.hole_size = hole_size
        self.sigma = sigma

    def remove_small_objects(self):
        processed_mask = np.zeros_like(self.mask)
        labeled_mask, num_labels = label(self.mask)
        sizes = np.bincount(labeled_mask.ravel())

        for cls in np.unique(self.mask):
            if cls == 0:  # Skip background if coded as 0
                continue

            class_mask = (self.mask == cls)
            labeled_class_mask, _ = label(class_mask)
            class_sizes = np.bincount(labeled_class_mask.ravel())
            class_mask_sizes = class_sizes >= self.min_size
            class_mask_sizes[0] = 0

            valid_objects = class_mask_sizes[labeled_class_mask]
            processed_mask[valid_objects] = cls

        self.mask = processed_mask

    def remove_small_holes(self):
        processed_mask = np.zeros_like(self.mask)
        
        for cls in np.unique(self.mask):
            if cls == 0:  # Skip background if coded as 0
                continue

            class_mask = (self.mask == cls)
            inverted_class_mask = ~class_mask
            filled_holes = self._remove_small_objects(inverted_class_mask)
            processed_mask[~filled_holes] = cls

        self.mask = processed_mask

    def _remove_small_objects(self, mask):
        labeled_mask, _ = label(mask)
        sizes = np.bincount(labeled_mask.ravel())
        mask_sizes = sizes >= self.hole_size
        mask_sizes[0] = 0
        return mask_sizes[labeled_mask]

    def smooth(self):
        smoothed_mask = np.zeros_like(self.mask, dtype=float)

        for cls in np.unique(self.mask):
            if cls == 0:  # Skip background if coded as 0
                continue

            class_mask = (self.mask == cls).astype(float)
            smoothed_class_mask = gaussian_filter(class_mask, sigma=self.sigma)
            smoothed_mask[smoothed_class_mask > 0.5] = cls

        self.mask = smoothed_mask.astype(self.mask.dtype)

    def morphological_operations(self):
        if self.mask.ndim != 2:
            raise ValueError("Mask must be a 2D array for morphological operations.")

        unique_classes = np.unique(self.mask)
        processed_mask = np.zeros_like(self.mask)

        structure = generate_binary_structure(2, 1)

        for cls in unique_classes:
            if cls == 0:  # Skip background if coded as 0
                continue

            class_mask = (self.mask == cls)
            class_mask = binary_erosion(class_mask, structure=structure)
            class_mask = binary_dilation(class_mask, structure=structure)

            processed_mask[class_mask] = cls

        self.mask = processed_mask

    def process(self):
        self.remove_small_objects()
        self.remove_small_holes()
        self.smooth()
        self.morphological_operations()
        return self.mask.astype(np.uint8)

class HyperspectralClassifier:
    def __init__(self, image_path, mask_path, classifier='rf'):
        self.image_path = image_path
        self.mask_path = mask_path
        self.X = None
        self.y = None
        self.height = None
        self.width = None
        self.n_components  = 20
        self.classifier = classifier
        if classifier == 'rf':
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        elif classifier == 'svm':
            self.model = SVC(kernel='linear', verbose=True, random_state=42)
        else:
            raise ValueError("Invalid classifier. Choose 'rf' or 'svm'.")
    
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

    def load_data(self):
        with rasterio.open(self.image_path) as src:
            self.hyperspectral_image = src.read()
            self.profile = src.profile
        n_bands, self.height, self.width = self.hyperspectral_image.shape
        self.all_X = self.hyperspectral_image.reshape(n_bands, -1).T
       

        with rasterio.open(self.mask_path) as mask_src:
            mask = mask_src.read(1)
        self.all_y = mask.flatten()

        # Omit the specific class
        mask_to_keep = self.all_y != 100
        self.X = self.all_X[mask_to_keep]
        self.y = self.all_y[mask_to_keep]

    def preprocess_labels(self):
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)

    def train_test_split(self):
        return train_test_split(self.X, self.y, test_size=0.5, random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Pixel-wise Accuracy: {accuracy * 100:.2f}%")

    def create_classification_mask(self):
        y_pred_full = []
        for i in tqdm(range(0, len(self.all_X), 1000), desc="Predicting"):
            chunk = self.all_X[i:i+1000]
            y_pred_chunk = self.model.predict(chunk)
            y_pred_full.extend(y_pred_chunk)
        preprocessed_mask = np.array(y_pred_full).reshape(self.height, self.width)

        # Post-process the classification mask
        processor = MaskProcessor(preprocessed_mask)
        post_processed_mask = processor.process()
        return post_processed_mask,preprocessed_mask

    def save_classification_mask(self, classification_mask, classification_file):
        profile = self.profile
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

        with rasterio.open(classification_file, 'w', **profile) as dst:
            dst.write(classification_mask.astype(rasterio.uint8), 1)

    def save_model(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, filepath):
        with open(filepath, 'rb') as file:
            self.model = pickle.load(file)

def classifypixel(hyperspectral_image, mask, classification_file, train=False, classifier_prefix='rf'):
    if train:
        classifier = HyperspectralClassifier(hyperspectral_image, mask, classifier=classifier_prefix)
        classifier.load_data()
        X_train, X_test, y_train, y_test = classifier.train_test_split()
        classifier.train_model(X_train, y_train)
        classifier.save_model(f'trainedmodel2_{classifier_prefix}')
        classifier.evaluate_model(X_test, y_test)
        post_processed_mask,preprocessed_mask = classifier.create_classification_mask()
        print("Classification mask created.")
        classifier.save_classification_mask(preprocessed_mask, classification_file+"_pre.tif")
        classifier.save_classification_mask(post_processed_mask, classification_file+"_post.tif")

    else:
        classifier = HyperspectralClassifier(hyperspectral_image, mask, classifier=classifier_prefix)
        classifier.load_data()
        classifier.load_model(f'trainedmodel2_{classifier_prefix}')
        post_processed_mask,preprocessed_mask = classifier.create_classification_mask()
        print("Classification mask created.")
        classifier.save_classification_mask(preprocessed_mask, classification_file+"_pre.tif")
        classifier.save_classification_mask(post_processed_mask, classification_file+"_post.tif")
