import tensorflow as tf
from tensorflow import keras    #I imported these like this instead of the way its done in notebook because vs code gives error
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing
x_train_reshape = np.reshape(x_train, (60000, 28, 28, 1))  # Add channel dimension
x_test_reshape = np.reshape(x_test, (10000, 28, 28, 1))
x_train_norm = x_train_reshape.astype('float32') / 255.0  # Normalize
x_test_norm = x_test_reshape.astype('float32') / 255.0
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # Rotate images randomly by up to 10 degrees
    width_shift_range=0.1,  # Shift images horizontally by up to 10% of the width
    height_shift_range=0.1,  # Shift images vertically by up to 10% of the height
    shear_range=0.1,  # Apply shear transformation by up to 10%
    zoom_range=0.1,  # Zoom in/out by up to 10%
    horizontal_flip=False,  # Do not flip horizontally
    vertical_flip=False  # Do not flip vertically
)
datagen.fit(x_train_norm)

# Model
model = Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training with data augmentation
history = model.fit(datagen.flow(x_train_norm, y_train_encoded, batch_size=32),
                    epochs=3,
                    validation_data=(x_test_norm, y_test_encoded))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set accuracy:', test_accuracy * 100)

# Make predictions
preds = model.predict(x_test_norm)

# Display 25 values from the test set along with predictions
plt.figure(figsize=(12, 12))
start_index = 0

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    pred = np.argmax(preds[start_index + i])
    gt = y_test[start_index + i]

    col = 'g'

    if pred != gt:
        col = 'r'

    plt.xlabel("i={}, pred={}, gt={}".format(start_index + i, pred, gt), color=col)
    plt.imshow(x_test[start_index + i], cmap='binary')
plt.show()
