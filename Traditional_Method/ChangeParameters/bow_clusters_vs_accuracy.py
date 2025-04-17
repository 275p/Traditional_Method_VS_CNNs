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
# Step 3: Extract Dense SIFT (Fixed Step Size)
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
# Step 4: Run Experiment with Varying n_clusters
# -------------------
def run_bow_experiment(n_clusters):
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

    clf = SVC(kernel='rbf', C=10, gamma='scale')
    clf.fit(X_train_bow, y_train_filtered)
    y_pred = clf.predict(X_test_bow)
    acc = accuracy_score(y_test_filtered, y_pred)
    return acc

# -------------------
# Step 5: Loop & Plot
# -------------------
cluster_counts = [100, 200, 300, 400, 500]
accuracies = []

for n in cluster_counts:
    print(f"Running experiment with {n} clusters")
    acc = run_bow_experiment(n)
    accuracies.append(acc)

plt.figure(figsize=(8, 6))
plt.plot(cluster_counts, accuracies, marker='o', linestyle='-')
plt.title('Effect of BoW Cluster Count on Accuracy (Fashion-MNIST)')
plt.xlabel('Number of Clusters (BoW)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('bow_clusters_vs_accuracy.png')
print("Saved plot: bow_clusters_vs_accuracy.png")
