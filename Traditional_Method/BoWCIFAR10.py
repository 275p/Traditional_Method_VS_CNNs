import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.datasets import cifar10
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------
# Step 1: Load CIFAR-10 and Subsample
# -------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()

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
# Step 2: Preprocess - Resize and Grayscale
# -------------------
def preprocess_image(img):
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

X_train_gray = [preprocess_image(img) for img in X_train]
X_test_gray = [preprocess_image(img) for img in X_test]

# -------------------
# Step 3: SIFT Feature Extraction with Label Tracking
# -------------------
sift = cv2.SIFT_create()

def extract_dense_sift_descriptors(images, labels, step_size=12):
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

train_descriptors_list, y_train_filtered = extract_dense_sift_descriptors(X_train_gray, y_train)

# -------------------
# Step 4: Create Codebook with KMeans
# -------------------
all_descriptors = np.vstack(train_descriptors_list)
n_clusters = 400
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
# Step 6: Test Set - Extract & Filter
# -------------------
test_descriptors_list, y_test_filtered = extract_dense_sift_descriptors(X_test_gray, y_test)
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

cifar10_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Output classification report
print("Classification results for CIFAR-10 (SVM with BoW):")
print(classification_report(y_test_filtered, y_pred, target_names=cifar10_labels))

# -------------------
# Step 10: Calculate Class-wise Accuracy
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

# Calculate class-wise accuracy
classwise_accuracy = calculate_classwise_accuracy(y_test_filtered, y_pred)

# -------------------
# Step 11: Plot Class-wise Accuracy Histogram
# -------------------
plt.figure(figsize=(8, 6))
plt.bar(np.arange(10), classwise_accuracy, tick_label=cifar10_labels)
plt.title('Class-wise Accuracy for CIFAR-10 (SVM with BoW)')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(True, axis='y')

# Save the figure
plt.tight_layout()
plt.savefig('classwise_accuracy_histogram.png')

print("Class-wise accuracy histogram saved as classwise_accuracy_histogram.png")
