import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


def relu(x):
    return x * (x > 0)


def d_relu(x):
    return 1 * (x > 0)


class neural_network:

    def __init__(self, nodes):
        self.input_dim = nodes[0]
        self.HL01_dim = nodes[1]
        self.HL02_dim = nodes[2]
        self.output_dim = nodes[3]

        self.W1 = 2 * (np.random.rand(self.input_dim, self.HL01_dim) -1)
        self.W2 = 2 * (np.random.rand(self.HL01_dim, self.HL02_dim) -1)
        self.W3 = 2 * (np.random.rand(self.HL02_dim, self.output_dim) -1)

        self.B1 = 2 * (np.random.rand(1, self.HL01_dim))
        self.B2 = 2 * (np.random.rand(1, self.HL02_dim))
        self.B3 = 2 * (np.random.rand(1, self.output_dim))

    def forward_pass(self, input):
        self.HL01_out = sigmoid(np.add(np.matmul(input, self.W1), self.B1))
        self.HL02_out = sigmoid(np.add(np.matmul(self.HL01_out, self.W2), self.B2))
        self.output = sigmoid(np.add(np.matmul(self.HL02_out, self.W3), self.B3))

        return self.output

    def backward_pass(self, train_data_X, train_data_Y, iterations, learning_rate):
        for j in range(iterations):
            self.forward_pass(train_data_X)

            error = np.sum(np.square(self.output - train_data_Y))
            print(error)

            output_error = self.output - train_data_Y
            output_deltas = output_error * d_sigmoid(self.output)

            self.W3 -= np.dot(self.HL02_out.T, output_deltas) * learning_rate
            self.B3 -= np.sum(output_deltas, axis=0, keepdims=True) * learning_rate

            HL02_error = np.dot(output_deltas, self.W3.T)
            HL02_deltas = HL02_error * d_sigmoid(self.HL02_out)

            self.W2 -= np.dot(self.HL01_out.T, HL02_deltas) * learning_rate
            self.B2 -= np.sum(HL02_deltas, axis=0, keepdims=True) * learning_rate

            HL01_error = np.dot(HL02_deltas, self.W2.T)
            HL01_deltas = HL01_error * d_sigmoid(self.HL01_out)

            self.W1 -= np.dot(train_data_X.T, HL01_deltas) * learning_rate
            self.B1 -= np.sum(HL01_deltas, axis=0, keepdims=True) * learning_rate
