import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

    def remove_small_objects(self):
        processed_mask = np.zeros_like(self.mask)
        
        for cls in np.unique(self.mask):
            if cls == 0:
                continue
                
            class_mask = (self.mask == cls)
            labeled_mask, _ = label(class_mask)
            sizes = np.bincount(labeled_mask.ravel())
            mask_sizes = sizes >= self.min_size
            mask_sizes[0] = 0
            processed_mask[mask_sizes[labeled_mask]] = cls
            
        self.mask = processed_mask
        return self.mask

    def remove_small_holes(self):
        processed_mask = np.zeros_like(self.mask)
        
        for cls in np.unique(self.mask):
            if cls == 0:
                continue
                
            class_mask = (self.mask == cls)
            holes = ~class_mask
            labeled_holes, _ = label(holes)
            sizes = np.bincount(labeled_holes.ravel())
            valid_holes = sizes >= self.hole_size
            valid_holes[0] = 0
            processed_mask[~valid_holes[labeled_holes]] = cls
            
        self.mask = processed_mask
        return self.mask

    def smooth(self):
        smoothed = np.zeros_like(self.mask, dtype=float)
        
        for cls in np.unique(self.mask):
            if cls == 0:
                continue
                
            class_mask = (self.mask == cls).astype(float)
            smoothed_class = gaussian_filter(class_mask, sigma=self.sigma)
            smoothed[smoothed_class > 0.5] = cls
            
        self.mask = smoothed.astype(self.mask.dtype)
        return self.mask

    def process(self):
        try:
            self.remove_small_objects()
            self.remove_small_holes()
            self.smooth()
            return self.mask.astype(np.uint8)
        except Exception as e:
            print(f"Error during mask processing: {str(e)}")
            return self.mask.astype(np.uint8)

class HyperspectralClassifier:
    def __init__(self, classifier_type='rf'):
        self.X = []
        self.y = []
        self.profiles = []
        
        if classifier_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42)
        else:
            raise ValueError("Supported classifiers: 'rf' or 'svm'")
            
    def add_data(self, image_path, mask_path):
        try:
            with rasterio.open(image_path) as src:
                image = src.read()
                self.profiles.append(src.profile)
                
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                
            X = image.reshape(image.shape[0], -1).T
            y = mask.ravel()
            
            valid_mask = y != 100
            self.X.extend(X[valid_mask])
            self.y.extend(y[valid_mask])
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            
    def prepare_data(self):
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
    def train(self):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.3, random_state=42
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model accuracy: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return None

    def save_model(self, path):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {str(e)}")

    def classify_image(self, image_path, output_path):
        try:
            with rasterio.open(image_path) as src:
                image = src.read()
                profile = src.profile.copy()
                
            X = image.reshape(image.shape[0], -1).T
            
            y_pred = []
            for i in tqdm(range(0, len(X), 1000)):
                batch = X[i:i+1000]
                pred = self.model.predict(batch)
                y_pred.extend(pred)
                
            mask = np.array(y_pred).reshape(image.shape[1:])
            
            processor = MaskProcessor(mask)
            processed_mask = processor.process()
            
            profile.update(dtype='uint8', count=1, compress='lzw')
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(processed_mask[np.newaxis, :, :])
                
            return processed_mask
            
        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return None

def classifypixel(image_paths, mask_paths, output_dir, train=True, classifier_prefix='rf'):
    classifier = HyperspectralClassifier(classifier_prefix)
    
    if train:
        for img_path, mask_path in zip(image_paths, mask_paths):
            classifier.add_data(img_path, mask_path)
        
        classifier.prepare_data()
        classifier.train()
        classifier.save_model(f'{output_dir}/model_{classifier_prefix}.pkl')
    else:
        classifier.load_model(f'{output_dir}/model_{classifier_prefix}.pkl')
    
    results = []
    for i, img_path in enumerate(image_paths):
        output_path = f'{output_dir}/classification_{i}.tif'
        mask = classifier.classify_image(img_path, output_path)
        results.append(mask)
        
    return results