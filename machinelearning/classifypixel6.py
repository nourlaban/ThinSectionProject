import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, label
import rasterio

class MaskProcessor:
    def __init__(self, mask, min_size=64, hole_size=64, sigma=1):
        self.mask = mask.astype(np.int32)
        self.min_size = min_size
        self.hole_size = hole_size
        self.sigma = sigma

    def process(self):
        try:
            self.remove_small_objects()
            self.remove_small_holes()
            self.smooth()
            return self.mask.astype(np.uint8)
        except Exception as e:
            print(f"Error during mask processing: {str(e)}")
            return self.mask.astype(np.uint8)

    # [Previous MaskProcessor methods remain the same]

class HyperspectralClassifier:
    def __init__(self, classifier='rf'):
        self.X_combined = []
        self.y_combined = []
        self.profiles = []
        
        if classifier == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier == 'svm':
            self.model = SVC(kernel='rbf', random_state=42)
        else:
            raise ValueError("Supported classifiers: 'rf' or 'svm'")

    def add_image_mask_pair(self, image_path, mask_path):
        try:
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
            
        except Exception as e:
            print(f"Error loading image-mask pair: {str(e)}")

    def prepare_data(self):
        self.X_combined = np.array(self.X_combined)
        self.y_combined = np.array(self.y_combined)

    def train(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            return True
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False

    def save_model(self, model_path):
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {str(e)}")

    def evaluate_model(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model accuracy: {accuracy:.4f}")
            return accuracy
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None

    def classify_single_image(self, image_path, output_path):
        try:
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
            
            processor = MaskProcessor(preprocessed_mask)
            post_processed_mask = processor.process()
            
            return post_processed_mask, preprocessed_mask, profile
            
        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return None, None, None

def train_classifier(image_mask_pairs, resPath, classifier_prefix='rf'):
    """
    Train the classifier using the provided image-mask pairs
    """
    classifier = HyperspectralClassifier(classifier=classifier_prefix)
    
    # Add all training data
    for img_path, mask_path in image_mask_pairs:
        print(f"Loading {img_path} and {mask_path}")
        classifier.add_image_mask_pair(img_path, mask_path)
    
    # Prepare and train the model
    classifier.prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        classifier.X_combined, 
        classifier.y_combined, 
        test_size=0.3, 
        random_state=42
    )
    
    if classifier.train(X_train, y_train):
        model_path = os.path.join(resPath, f'trainedmodel2_{classifier_prefix}.pkl')
        classifier.save_model(model_path)
        accuracy = classifier.evaluate_model(X_test, y_test)
        return accuracy
    return None

def classify_images(image_mask_pairs, resPath, classifier_prefix='rf'):
    """
    Classify images using a trained model
    """
    classifier = HyperspectralClassifier(classifier=classifier_prefix)
    model_path = os.path.join(resPath, f'trainedmodel2_{classifier_prefix}.pkl')
    classifier.load_model(model_path)
    
    results = []
    for img_path, _ in image_mask_pairs:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        post_processed_mask, preprocessed_mask, profile = classifier.classify_single_image(
            img_path, 
            os.path.join(resPath, f'classification_{base_name}.tif')
        )
        
        if profile is not None:
            # Save the classification masks
            output_base = os.path.join(resPath, f"classification_{base_name}")
            profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
            
            # Save pre-processed mask
            pre_path = f"{output_base}_pre.tif"
            with rasterio.open(pre_path, 'w', **profile) as dst:
                dst.write(preprocessed_mask.astype(rasterio.uint8), 1)
            
            # Save post-processed mask
            post_path = f"{output_base}_post.tif"
            with rasterio.open(post_path, 'w', **profile) as dst:
                dst.write(post_processed_mask.astype(rasterio.uint8), 1)
                
            results.append({
                'image': img_path,
                'pre_processed': pre_path,
                'post_processed': post_path
            })
            
            print(f"Processed {base_name}")
    
    return results