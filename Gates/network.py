import numpy as np

# a simple network that predicts the output of a logic gate provided two inputs
class Network:
    def __init__(self, data):
        self.data = data
        self.learning_rate = 0.2
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.bias = np.random.randn()

        self.train()

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_p(self, z):
    # derivative of sigmoid function
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

    # def cost(self, z, target):
    #     return np.square(self.sigmoid(z) - target)

    # # derivative of cost function
    # def cost_p(self, z, target):
    #     return 2 * (self.sigmoid(z) - target)

    # training loop for neural network
    def train(self):
        for i in range(50000):
            rand_i = np.random.randint(len(self.data))
            point = self.data[rand_i]

            z = point[0] * self.w1 + point[1] * self.w2 + self.bias
            pred = self.sigmoid(z)
            target = point[2]
            cost = np.square(pred - target)

            dcost_dpred = 2 * (pred - target)
            dpred_dz = self.sigmoid_p(z)
            
            dz_dw1 = point[0]
            dz_dw2 = point[1]
            dz_db = 1

            dcost_dz = dcost_dpred * dpred_dz
            dcost_dw1 = dcost_dz * dz_dw1
            dcost_dw2 = dcost_dz * dz_dw2
            dcost_db = dcost_dz * dz_db

            self.w1 -= self.learning_rate * dcost_dw1
            self.w2 -= self.learning_rate * dcost_dw2
            self.bias -= self.learning_rate * dcost_db

    def predict(self, a, b):
        z = self.w1 * a + self.w2 * b + self.bias
        pred = self.sigmoid(z)
        return pred