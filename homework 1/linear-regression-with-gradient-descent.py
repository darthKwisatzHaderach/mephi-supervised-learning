import numpy as np
import pandas as pd


class LinearRegression():

    def __init__(self, learning_rate, iterations):
        self.theta = None
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        self.theta = np.zeros(2)
        self.X_b = np.c_[np.ones(len(X)), X]
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()

        self.bias = self.theta[0]
        self.weights = self.theta[1]
        return self

    def update_weights(self):
        m = self.X_b.shape[0]
        grad = (2 / m) * self.X_b.T.dot(self.X_b.dot(self.theta) - self.Y)
        self.theta = self.theta - self.learning_rate * grad

    def predict(self, X):
        X = np.atleast_1d(X)
        X_new = np.c_[np.ones(len(X)), X]

        return X_new.dot(self.theta)

df = pd.read_csv('salary_data.csv')

X = df.iloc[:,:-1].values
Y = df.iloc[:,1].values

model = LinearRegression(iterations = 1000, learning_rate = 0.01)
model.fit(X, Y)

Y_pred = model.predict(X)

print(np.round(model.weights, 2))
print(np.round(model.bias, 2))