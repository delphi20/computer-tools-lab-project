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
x_train_reshape = np.reshape(x_train, (60000, 784))
x_test_reshape = np.reshape(x_test, (10000, 784))
x_mean = np.mean(x_train_reshape)
x_std = np.std(x_train_reshape)
epsilon = 1e-10
x_train_norm = (x_train_reshape - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshape - x_mean) / (x_std + epsilon)
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Modify model to reduce accuracy
model = Sequential([
    Dense(32, activation='sigmoid', input_shape=(784,)),  # Reduced neurons and changed activation to sigmoid
    Dense(32, activation='sigmoid'),  # Reduced neurons and changed activation to sigmoid
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',  # Using Adam optimizer
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(x_train_norm, y_train_encoded, epochs=3, batch_size=32)  # Reduced epochs and batch size

# Evaluate the model
_, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set accuracy: ', accuracy * 100)

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
