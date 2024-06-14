import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def sse(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)*0.5

class FFNN:
    def __init__(self, input_layer, hidden_layer, output_layer):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        # Initialize weights and biases randomly
        self.W1 = np.random.rand(self.input_layer, self.hidden_layer)     # Weights between input layer and hidden layer
        self.W2 = np.random.rand(self.hidden_layer, self.output_layer)    # Weights between hidden layer and output layer
        self.B1 = np.random.rand(1, self.hidden_layer)                    # Bias for hidden layer
        self.B2 = np.random.rand(1, self.output_layer)                    # Bias for output layer
        self.learning_rate = 0.01

    def train(self, inp_train, out_train, learning_rate=0.1, epochs=1000):
        
        for epoch in range(epochs):
            error = 0
            
            for inp, true_out in zip(inp_train, out_train):
                # Forward Propagation
                h1 = sigmoid(np.dot(inp, self.W1) + self.B1)            # Hidden layer
                h1 = h1.flatten()
                y = sigmoid(np.dot(h1, self.W2) + self.B2)              # Output layer
                y = y.flatten()

                # Error
                error += sse(true_out, y)

                # Backward Propagation

                dW1 = np.zeros((self.input_layer, self.hidden_layer))
                dW2 = np.zeros((self.hidden_layer, self.output_layer))
                dB1 = np.zeros(self.hidden_layer)
                dB2 = np.zeros(self.output_layer)

                # Compute gradients
                for k in range(self.output_layer):
                    dB2[k] = (y[k] - true_out) * sigmoid_derivative(y[k])
                
                for j in range(self.hidden_layer):
                    for k in range(self.output_layer):
                        dW2[j][k] = dB2[k] * h1[j]

                for j in range(self.hidden_layer):
                    dB1[j] = np.sum([dB2[k] * self.W2[j][k] for k in range(self.output_layer)]) * sigmoid_derivative(h1[j])
                
                for i in range(self.input_layer):
                    for j in range(self.hidden_layer):
                        dW1[i][j] = dB1[j] * inp[i]

                # Update weights and biases
                self.W1 -= learning_rate * dW1
                self.W2 -= learning_rate * dW2
                self.B1 -= learning_rate * dB1
                self.B2 -= learning_rate * dB2

                error /= inp_train.shape[0]
                print(f"Epoch: {epoch}, Error: {error}")


        
