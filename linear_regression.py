import numpy as np

class LinearRegression:

    # Initialize the Linear Regression model with separate intercept term
    # lr: learning rate, n_iters: number of iterations for gradient descent
    # coeff: coefficients for features (b1, b2.....), intercept: bias term (b0)

    def __init__(self, lr = 0.001, n_iters=1000, l2 = 0, l1 = 0):
        self.lr = lr
        self.n_iters = n_iters
        self.coeff = None
        self.intercept = None
        self.l2 = l2
        self.l1 = l1

    def calc_params(self, X, y):
        n_samples, n_features = X.shape
        self.coeff = np.zeros(n_features)
        self.intercept = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.coeff) + self.intercept
            # take transpose of X to match dimensions for dot product
            dw = (1/n_samples) * ( np.dot(X.T, (y_pred-y)) + self.l2 * self.coeff + self.l1 * np.sign(self.coeff) )
            db = (1/n_samples) * np.sum(y_pred-y)
            self.coeff = self.coeff - self.lr * dw
            self.intercept = self.intercept - self.lr * db
            

            self.coeff = self.coeff - self.lr * dw
            self.intercept = self.intercept - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.coeff) + self.intercept
        

        return y_pred
    
class LinearRegression_1:

    # Initialize the Linear Regression model with separate intercept term
    # lr: learning rate, n_iters: number of iterations for gradient descent
    # coeff: coefficients for features (b1, b2.....), intercept: bias term (b0)

    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.coeff = None

    def calc_params(self, X, y):
        n_samples, n_features = X.shape
        self.coeff = np.zeros(n_features + 1) # +1 for intercept

        for _ in range(self.n_iters):

            X_b = np.c_[np.ones((n_samples, 1)), X]
            y_pred = np.dot(X_b, self.coeff)
            dw = (1/n_samples) * np.dot(X_b.T, (y_pred-y))
            
            self.coeff = self.coeff - self.lr * dw


    def predict(self, X):
        n_samples  = X.shape[0]
        X_b = np.c_[np.ones((n_samples, 1)), X]
        y_pred = np.dot(X_b, self.coeff)

        return y_pred