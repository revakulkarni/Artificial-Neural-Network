#Write a python program to design a Hopfield Network which stores 4 vectors
#Hebbian Rule

import numpy as np 
class HopfieldNetwork:
    def __init__(self, n_neurons): 
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern) 
            np.fill_diagonal(self.weights, 0)

    def predict(self, pattern):
        energy = -0.5 * ((pattern @ self.weights) @ pattern) 
        return np.sign((pattern @ self.weights) + energy)


patterns = np.array([
                [1, 1, -1, -1],
                [-1, -1, 1, 1],
                [1, -1, 1, -1],
                [-1, 1, -1, 1]
                ])
n_neurons = 4


network = HopfieldNetwork(n_neurons) 

network.train(patterns)

for pattern in patterns:
    prediction = network.predict(pattern) 
    print('Input pattern:', pattern) 
    print('Predicted pattern:', prediction)


