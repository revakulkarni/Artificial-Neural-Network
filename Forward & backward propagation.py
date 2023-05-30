import numpy as np 
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size): 
        self.W1 = np.random.randn(input_size, hidden_size) 
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) 
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x): 
        return x * (1 - x)

    def forward_propagation(self, X): 
        self.input = np.dot(X, self.W1) + self.b1 
        self.hid_input = self.sigmoid(self.input) #a1
        self.out_input = np.dot(self.hid_input, self.W2) + self.b2 
        self.final_output = self.sigmoid(self.out_input)
        return self.final_output

    def backward_propagation(self, X, y, y_hat): 
        self.error = y - final_output
        self.delta2 = self.error * self.sigmoid_derivative(final_output) 
        self.a1_error = self.delta2.dot(self.W2.T) 
        self.delta1 = self.a1_error * self.sigmoid_derivative(self.a1) 
        self.W2 += self.a1.T.dot(self.delta2)
        self.b2 += np.sum(self.delta2, axis=0, keepdims=True) 
        self.W1 += X.T.dot(self.delta1)
        self.b1 += np.sum(self.delta1, axis=0)

    def train(self, X, y, epochs, learning_rate): 
        for i in range(epochs):
            final_output = self.forward_propagation(X)
            self.backward_propagation(X, y, final_output6) 
            if i % 100 == 0:
                print("Error at epoch", i, ":", np.mean(np.abs(self.error)))

    def predict(self, X):
        # Make predictions for a given set of inputs 
        return self.forward_propagation(X)


# Define the input and output datasets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network with 2 input neurons, 4 neurons in the hidden layer, and 1 output neuron
nn = NeuralNetwork(2, 4, 1)

# Train the neural network on the input and output datasets for 10000 epochs with a learning rate of 0.1
nn.train(X, y, learning_rate=0.1, epochs=10000)

# Use the trained neural network to make predictions on the same input dataset 
predictions = nn.predict(X)

# Print the predictions 
print(predictions)
