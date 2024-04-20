"""
This code will train the model for 3 epochs first and display accuracy for each epoch,
this is still only with the training sets.

Then it'll evaluate the model with the test sets.
Then it'll roll the predictions into an array called preds using
model.predict.

Then finally it'll display 25 values from the test set, along with the predictions
the model made for them

"""

import tensorflow as tf
from tensorflow import keras    #I imported these like this instead of the way its done in notebook because vs code gives error
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()    #Load data into arrays, I removed all the checking and stuff

y_train_encoded = to_categorical(y_train)   #One hot encoding the y arrays
y_test_encoded = to_categorical(y_test)


import numpy as np

x_train_reshape = np.reshape(x_train, (60000, 784))     #Reshaping the arrays, multiplying 28 by 28 to give 784 dimensional vector
x_test_reshape = np.reshape(x_test, (10000, 784))

#Normalising the data in the x arrays
x_mean = np.mean(x_train_reshape)   
x_std = np.std(x_train_reshape)
epsilon = 1e-10 #Small val so  we don't get zero in the denominator
x_train_norm = (x_train_reshape - x_mean)/(x_std + epsilon)
x_test_norm = (x_test_reshape - x_mean)/(x_std + epsilon)


#Importing and making the model
model = Sequential([
    Dense(128, activation = 'relu', input_shape = (784,)), #No input layer required
    Dense(128, activation='relu'),  #2 hidden layers with 128 nodes, also i hope you remember relu
    Dense(10, activation = 'softmax')   #output layer, softmax gives probability for which switch will be on in the y array
])


# compiling the model, dk the optimizer algorithims or loss and metrics rn
model.compile(
    optimizer = 'sgd',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#model.summary() (Intentionally commented out)

print('\n\n')

model.fit(x_train_norm, y_train_encoded, epochs = 3) #Training the model for three epochs (iterations)

print('\n\n')

_, accuracy = model.evaluate(x_test_norm, y_test_encoded)   #Evaluating the model using the test sets finally
#print('Test set accuracy: ', accuracy * 100)

preds = model.predict(x_test_norm)  #making predictions
#print('Shape of preds: ', preds.shape)


#All this is just to display the images and their predicted values
plt.figure(figsize=(12,12))
start_index = 0

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index+i])
    gt = y_test[start_index+i]
    
    col = 'g'
    
    if pred != gt:
        col = 'r'
    
    plt.xlabel("i={}, pred={}, gt={}".format(start_index+i, pred, gt), color = col)
    plt.imshow(x_test[start_index+i], cmap = 'binary')
plt.show()
    
#theres one last cell in the notebook for seeing what probabilities the model gave for what values, you can use that yourself
#if you want

