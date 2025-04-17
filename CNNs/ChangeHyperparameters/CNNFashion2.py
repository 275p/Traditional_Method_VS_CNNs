import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np

# Use non-GUI backend for matplotlib to prevent Qt errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape and normalize
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model for Fashion-MNIST
fashion_model_2 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(), input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU()),  # 更改为LeakyReLU
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU()),  # 更改为LeakyReLU
    layers.Flatten(),
    layers.Dense(64, activation=tf.keras.layers.LeakyReLU()),  # 更改为LeakyReLU
    layers.Dense(10, activation='softmax')
])

# Compile the model
fashion_model_2.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
fashion_history = fashion_model_2.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Predict on test set
fashion_y_pred = fashion_model_2.predict(x_test)
fashion_y_pred_classes = np.argmax(fashion_y_pred, axis=1)

# Print classification report
print("Classification results for Fashion-MNIST (CNN):")
print(classification_report(np.argmax(y_test, axis=1), fashion_y_pred_classes))

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(fashion_history.history['accuracy'], label='Train Accuracy')
plt.plot(fashion_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fashion-MNIST CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(fashion_history.history['loss'], label='Train Loss')
plt.plot(fashion_history.history['val_loss'], label='Validation Loss')
plt.title('Fashion-MNIST CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()

# Save plot
plt.savefig('fashion_mnist_cnn_training2.png')
print("Training process visualization saved as fashion_mnist_cnn_training2.png")
