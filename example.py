import nn
import numpy as np

"""
creating the neural network object with following dimensions:
2 input nodes
5 hidden layer 1 nodes
5 hidden layer 2 nodes
1 output node
"""

nn = nn.neural_network([2,5,5,1])

#Creating sample data for it to learn from, in this case I will make it learn on an AND table

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])

#Training the neural network on the dummy dataset

nn.backward_pass(x,y,100000,0.1)

#Showing test results on a couple of sample data

x_test = np.array([[0,1],[1,1]])
print(nn.forward_pass(x_test))


