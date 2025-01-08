

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
    def __init__(self, classifier='rf'):
        self.X_combined = []
        self.y_combined = []
        self.profiles = []
        self.n_components = 20
        
        if classifier == 'rf':
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        elif classifier == 'svm':
            self.model = SVC(kernel='linear', verbose=True, random_state=42)
        else:
            raise ValueError("Invalid classifier. Choose 'rf' or 'svm'.")

    def add_image_mask_pair(self, image_path, mask_path):
        with rasterio.open(image_path) as src:
            hyperspectral_image = src.read()
            self.profiles.append(src.profile)
            height, width = src.height, src.width
        
        n_bands = hyperspectral_image.shape[0]
        X = hyperspectral_image.reshape(n_bands, -1).T

        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)
        y = mask.flatten()

        # Omit the specific class (100)
        mask_to_keep = y != 100
        X = X[mask_to_keep]
        y = y[mask_to_keep]

        self.X_combined.extend(X)
        self.y_combined.extend(y)

    def prepare_data(self):
        self.X_combined = np.array(self.X_combined)
        self.y_combined = np.array(self.y_combined)
        
    def train_test_split(self):
        return train_test_split(self.X_combined, self.y_combined, test_size=0.5, random_state=42)

    # ... (keep other methods from the original class) ...

    def classify_single_image(self, image_path, output_path):
        with rasterio.open(image_path) as src:
            hyperspectral_image = src.read()
            profile = src.profile
            height, width = src.height, src.width
        
        n_bands = hyperspectral_image.shape[0]
        X = hyperspectral_image.reshape(n_bands, -1).T

        y_pred_full = []
        for i in tqdm(range(0, len(X), 1000), desc="Predicting"):
            chunk = X[i:i+1000]
            y_pred_chunk = self.model.predict(chunk)
            y_pred_full.extend(y_pred_chunk)

        preprocessed_mask = np.array(y_pred_full).reshape(height, width)
        
        # Post-process the classification mask
        processor = MaskProcessor(preprocessed_mask)
        post_processed_mask = processor.process()
        
        return post_processed_mask, preprocessed_mask, profile

def classifypixel(image_mask_pairs, classification_file, train=False, classifier_prefix='rf'):
    classifier = HyperspectralClassifier(classifier=classifier_prefix)
    
    if train:
        # Add all training data
        for img_path, mask_path in image_mask_pairs:
            print(f"Loading {img_path} and {mask_path}")
            classifier.add_image_mask_pair(img_path, mask_path)
        
        # Prepare and train the model
        classifier.prepare_data()
        X_train, X_test, y_train, y_test = classifier.train_test_split()
        classifier.model(X_train, y_train)
        classifier.save_model(f'trainedmodel2_{classifier_prefix}')
        classifier.evaluate_model(X_test, y_test)
    else:
        classifier.load_model(f'trainedmodel2_{classifier_prefix}')
    
    # Classify each image
    for idx, (img_path, _) in enumerate(image_mask_pairs):
        post_processed_mask, preprocessed_mask, profile = classifier.classify_single_image(img_path, classification_file)
        
        # Save the classification masks with unique names
        output_base = f"{classification_file}_{idx+1}"
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
        
        with rasterio.open(f"{output_base}_pre.tif", 'w', **profile) as dst:
            dst.write(preprocessed_mask.astype(rasterio.uint8), 1)
        
        with rasterio.open(f"{output_base}_post.tif", 'w', **profile) as dst:
            dst.write(post_processed_mask.astype(rasterio.uint8), 1)

