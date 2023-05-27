import numpy as np

class XORNetwork: 
    def __init__(self):
        # Initialize the weights and biases randomly 
        self.W1 = np.random.randn(2, 2)
        self.b1 = np.random.randn(2) 
        self.W2 = np.random.randn(2, 1) 
        self.b2 = np.random.randn(1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x): 
        return x * (1 - x)

    def forward(self, X):
        # Perform the forward pass
        self.z1 = np.dot(X, self.W1) + self.b1 
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2 
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output): # Perform the backward pass 
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        self.z1_error = self.output_delta.dot(self.W2.T)
        self.z1_delta = self.z1_error * self.sigmoid_derivative(self.a1)
        self.W1 += X.T.dot(self.z1_delta) 
        self.b1 += np.sum(self.z1_delta, axis=0)
        self.W2 += self.a1.T.dot(self.output_delta) 
        self.b2 += np.sum(self.output_delta, axis=0)

    def train(self, X, y, epochs):
        # Train the network for a given number of epochs 
        for i in range(epochs):
            output = self.forward(X) 
            self.backward(X, y, output)

    def predict(self, X):
        # Make predictions for a given set of inputs 
        return self.forward(X)

# Create a new XORNetwork instance 
xor_nn = XORNetwork()

# Define the input and output datasets for XOR 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the network for 10000 epochs 
xor_nn.train(X, y, epochs=10000)

# Make predictions on the input dataset 
predictions = xor_nn.predict(X)

# Print the predictions 
print(predictions)
