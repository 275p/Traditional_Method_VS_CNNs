import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Residual block definition
def residual_block(x, filters, downsample=False):
    shortcut = x
    stride = 2 if downsample else 1

    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# Build ResNet-style model
def build_resnet(input_shape=(28, 28, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    x = residual_block(x, 32)
    x = residual_block(x, 32)

    x = residual_block(x, 64, downsample=True)
    x = residual_block(x, 64)

    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x)
    return model

# Load Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build and compile model
resnet_fashion_model = build_resnet()
resnet_fashion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
resnet_fashion_history = resnet_fashion_model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_test, y_test))

# Predict and evaluate
fashion_y_pred = resnet_fashion_model.predict(x_test)
fashion_y_pred_classes = np.argmax(fashion_y_pred, axis=1)

print("Classification results for Fashion-MNIST (ResNet-style CNN):")
print(classification_report(np.argmax(y_test, axis=1), fashion_y_pred_classes))

# Plot training process
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(resnet_fashion_history.history['accuracy'], label='Train Accuracy')
plt.plot(resnet_fashion_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('ResNet-Style Model Accuracy (Fashion-MNIST)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(resnet_fashion_history.history['loss'], label='Train Loss')
plt.plot(resnet_fashion_history.history['val_loss'], label='Validation Loss')
plt.title('ResNet-Style Model Loss (Fashion-MNIST)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('fashion_resnet_training_visualization.png')
print("Training process visualization saved as fashion_resnet_training_visualization.png")
