#Write a python program for bidirectional associative memory with two pairs of vectors

import numpy as np

# define two pairs of vectors 
x1 = np.array([1, 1, 1, -1])
y1 = np.array([1, -1])
x2 = np.array([-1, -1, 1, 1]) 
y2 = np.array([-1, 1])

# compute weight matrix W
W = np.outer(y1, x1) + np.outer(y2, x2)

# define BAM function 
def bam(x):
    y = W @ x
    return np.where(y>=0, 1, -1)

# test BAM with inputs
x_test = np.array([1, -1, -1, -1]) 
y_test = bam(x_test)

# print output 
print("Input x: ", x_test) 
print("Output y: ", y_test)
