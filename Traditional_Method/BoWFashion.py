import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import fashion_mnist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------
# Step 1: Load Fashion-MNIST and Subsample
# -------------------
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

def subsample_by_class(X, y, samples_per_class=1000):
    X_sub, y_sub = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        selected = np.random.choice(idx, samples_per_class, replace=False)
        X_sub.extend(X[selected])
        y_sub.extend(y[selected])
    return np.array(X_sub), np.array(y_sub)

X_train, y_train = subsample_by_class(X_train, y_train, samples_per_class=1000)
X_test, y_test = subsample_by_class(X_test, y_test, samples_per_class=200)

# -------------------
# Step 2: Resize Images (since they're already grayscale)
# -------------------
def resize_image(img, size=128):
    return cv2.resize(img, (size, size))

X_train_resized = [resize_image(img) for img in X_train]
X_test_resized = [resize_image(img) for img in X_test]

# -------------------
# Step 3: Dense SIFT Extraction
# -------------------
sift = cv2.SIFT_create()

def extract_dense_sift_descriptors(images, labels, step_size=8):
    descriptors_list = []
    valid_labels = []
    for img, label in zip(images, labels):
        keypoints = [cv2.KeyPoint(x, y, step_size)
                     for y in range(0, img.shape[0], step_size)
                     for x in range(0, img.shape[1], step_size)]
        _, descriptors = sift.compute(img, keypoints)
        if descriptors is not None:
            descriptors_list.append(descriptors)
            valid_labels.append(label)
    return descriptors_list, np.array(valid_labels)

train_descriptors_list, y_train_filtered = extract_dense_sift_descriptors(X_train_resized, y_train)

# -------------------
# Step 4: Create Codebook with MiniBatchKMeans
# -------------------
all_descriptors = np.vstack(train_descriptors_list)
n_clusters = 300
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
kmeans.fit(all_descriptors)

# -------------------
# Step 5: Convert Images to BoW Vectors
# -------------------
def compute_bow_histograms(descriptor_list, kmeans):
    histograms = []
    for descriptors in descriptor_list:
        if descriptors is None:
            histogram = np.zeros(kmeans.n_clusters)
        else:
            words = kmeans.predict(descriptors)
            histogram, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
        histograms.append(histogram)
    return np.array(histograms)

X_train_bow = compute_bow_histograms(train_descriptors_list, kmeans)

# -------------------
# Step 6: Process Test Set
# -------------------
test_descriptors_list, y_test_filtered = extract_dense_sift_descriptors(X_test_resized, y_test)
X_test_bow = compute_bow_histograms(test_descriptors_list, kmeans)

# -------------------
# Step 7: Normalize BoW Vectors
# -------------------
scaler = StandardScaler()
X_train_bow = scaler.fit_transform(X_train_bow)
X_test_bow = scaler.transform(X_test_bow)

# -------------------
# Step 8: Train SVM
# -------------------
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train_bow, y_train_filtered)

# -------------------
# Step 9: Predict & Evaluate
# -------------------
y_pred = svm.predict(X_test_bow)

fashion_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print("Classification results for Fashion-MNIST (SVM with BoW):")
print(classification_report(y_test_filtered, y_pred, target_names=fashion_labels))

# -------------------
# Step 10: Class-wise Accuracy
# -------------------
def calculate_classwise_accuracy(y_true, y_pred, num_classes=10):
    accuracy_per_class = []
    for i in range(num_classes):
        true_class = (y_true == i)
        pred_class = (y_pred == i)
        correct = np.sum(true_class & pred_class)
        total = np.sum(true_class)
        class_accuracy = correct / total if total > 0 else 0
        accuracy_per_class.append(class_accuracy)
    return accuracy_per_class

classwise_accuracy = calculate_classwise_accuracy(y_test_filtered, y_pred)

# -------------------
# Step 11: Plot Histogram
# -------------------
plt.figure(figsize=(8, 6))
plt.bar(np.arange(10), classwise_accuracy, tick_label=fashion_labels)
plt.title('Class-wise Accuracy for Fashion-MNIST (SVM with BoW)')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('fashionmnist_classwise_accuracy.png')

print("Class-wise accuracy histogram saved as fashionmnist_classwise_accuracy.png")
