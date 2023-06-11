#MNIST Handwritten Character Detection using PyTorch, Keras and Tensorflow

import tensorflow as tf
from keras.datasets import mnist 
from keras.models import Sequential
from keras.layers import Dense, Flatten 

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

#reshaping of input values
X_train = X_train / 255.0   #(28,28)
X_test = X_test / 255.0

# Define the model architecture 
model = Sequential([
    Flatten(input_shape=(28, 28)), 
    Dense(128, activation='relu'), 
    Dense(10, activation='softmax')
])

# Compile the model 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  #multi class classification problems
                metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64 )

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test) 
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
