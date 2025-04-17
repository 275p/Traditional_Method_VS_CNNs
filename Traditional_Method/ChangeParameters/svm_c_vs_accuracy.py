import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import fashion_mnist

# -------------------
# Step 1: Load and Subsample
# -------------------
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

def subsample_by_class(X, y, samples_per_class=500):
    X_sub, y_sub = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        selected = np.random.choice(idx, samples_per_class, replace=False)
        X_sub.extend(X[selected])
        y_sub.extend(y[selected])
    return np.array(X_sub), np.array(y_sub)

X_train, y_train = subsample_by_class(X_train, y_train)
X_test, y_test = subsample_by_class(X_test, y_test, samples_per_class=100)

# -------------------
# Step 2: Resize
# -------------------
def resize_image(img, size=128):
    return cv2.resize(img, (size, size))

X_train_resized = [resize_image(img) for img in X_train]
X_test_resized = [resize_image(img) for img in X_test]

# -------------------
# Step 3: Dense SIFT Extraction
# -------------------
step_size = 8
sift = cv2.SIFT_create()

def extract_dense_sift(images, labels):
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

train_descriptors, y_train_filtered = extract_dense_sift(X_train_resized, y_train)
test_descriptors, y_test_filtered = extract_dense_sift(X_test_resized, y_test)

# -------------------
# Step 4: Create BoW with fixed cluster number
# -------------------
n_clusters = 300
all_descriptors = np.vstack(train_descriptors)
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
kmeans.fit(all_descriptors)

def compute_bow(descriptor_list):
    histograms = []
    for desc in descriptor_list:
        words = kmeans.predict(desc)
        histogram, _ = np.histogram(words, bins=np.arange(n_clusters + 1))
        histograms.append(histogram)
    return np.array(histograms)

X_train_bow = compute_bow(train_descriptors)
X_test_bow = compute_bow(test_descriptors)

scaler = StandardScaler()
X_train_bow = scaler.fit_transform(X_train_bow)
X_test_bow = scaler.transform(X_test_bow)

# -------------------
# Step 5: Experiment with different C values
# -------------------
C_values = [0.01, 0.1, 1, 10, 100]
accuracies = []

for C in C_values:
    print(f"Training SVM with C={C}")
    clf = SVC(kernel='rbf', C=C, gamma='scale')
    clf.fit(X_train_bow, y_train_filtered)
    y_pred = clf.predict(X_test_bow)
    acc = accuracy_score(y_test_filtered, y_pred)
    accuracies.append(acc)

# -------------------
# Step 6: Plot
# -------------------
plt.figure(figsize=(8, 6))
plt.plot(C_values, accuracies, marker='o', linestyle='-')
plt.xscale('log')
plt.title('Effect of SVM C on Accuracy (Fashion-MNIST)')
plt.xlabel('C (SVM Regularization Parameter)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig('svm_c_vs_accuracy.png')
print("Saved plot: svm_c_vs_accuracy.png")
