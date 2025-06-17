from __future__ import print_function
import numpy as np
import pandas as pd

class MatrixFactorization(object):
    def __init__(self, Y, K, lam=0.1, Xinit=None, Winit=None,
                 learning_rate=0.5, max_iter=1000, print_every=100):
        self.Y = Y # this is the utility matrix
        self.K = K
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every
        self.n_users = int(np.max(Y[:, 0])) + 1
        self.n_items = int(np.max(Y[:, 1])) + 1
        self.n_ratings = Y.shape[0]
        self.X = np.random.randn(self.n_items, K) if Xinit is None else Xinit
        self.W = np.random.randn(K, self.n_users) if Winit is None else Winit
        self.b = np.random.randn(self.n_items) # item biases
        self.d = np.random.randn(self.n_users)
        
    def loss(self):
        L = 0
        for i in range(self.n_ratings):
            # user_id, item_id, rating
            n, m, rating = int(self.Y[i, 0]), int(self.Y[i, 1]), self.Y[i, 2]
            L += 0.5 * (self.X[m].dot(self.W[:, n]) + self.b[m] + self.d[n] - rating) ** 2  
            
        L /= self.n_ratings # number of ratings
        
        L_total = L + 0.5 * self.lam * (np.sum(self.X**2) + np.sum(self.W**2))
        
        return L_total
    
    def updateXb(self):
        for m in range(self.n_items):
            # obtain all users who rated item m and get the corresponding ratings
            ids = np.where(self.Y[:, 1] == m)[0] # row indices of items m
            user_ids, ratings = self.Y[ids, 0].astype(np.int32), self.Y[ids, 2]
            Wm, dm = self.W[:, user_ids], self.d[user_ids]
            for i in range(30):
                xm = self.X[m]
                error = xm.dot(Wm) + self.b[m] + dm - ratings
                grad_xm = error.dot(Wm.T) / self.n_ratings + self.lam * xm
                grad_bm = np.sum(error) / self.n_ratings
                self.X[m] -= self.learning_rate * grad_xm.reshape(-1)
                self.b[m] -= self.learning_rate * grad_bm
                
    def updateWd(self):
        for n in range(self.n_users):
            # obtain all items rated by user n and get the corresponding ratings
            ids = np.where(self.Y[:, 0] == n)[0]
            item_ids, ratings = self.Y[ids, 1].astype(np.int32), self.Y[ids, 2]
            Xn, bn = self.X[item_ids], self.b[item_ids]
            for i in range(30):
                wn = self.W[:, n]
                error = Xn.dot(wn) + bn + self.d[n] - ratings
                grad_wn = error.dot(Xn) / self.n_ratings + self.lam * wn
                grad_dn = np.sum(error) / self.n_ratings     
                self.W[:, n] -= self.learning_rate * grad_wn.reshape(-1)
                self.d[n] -= self.learning_rate * grad_dn
                
    def fit(self):
        for it in range(self.max_iter):
            self.updateWd()
            self.updateXb()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y)
                print('iter = %d, loss = %.4f, RMSE = %.4f' % (it + 1, self.loss(), rmse_train))
                
    def pred(self, user_id, item_id):
        user_id, item_id = int(user_id), int(item_id)
        pred = self.X[item_id, :].dot(self.W[:, user_id]) + self.b[item_id] + self.d[user_id]
        return max(0, min(pred, 5))
    
    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2]) ** 2     
        RMSE = np.sqrt(SE / n_tests)
        return RMSE
    

r_cols = ["user_id", "item_id", "rating", "timestamp"]
ratings_base = pd.read_csv("/kaggle/input/movie-lens-100k/ua.base", sep='\t', names=r_cols)
ratings_test = pd.read_csv("/kaggle/input/movie-lens-100k/ua.test", sep='\t', names=r_cols)

rate_train = ratings_base.values
rate_test = ratings_test.values

rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

# MATRIX FACTORIZATION
rs = MatrixFactorization(rate_train, K = 50, lam = .01, print_every=5, learning_rate=50,
max_iter = 30)
rs.fit()

# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print("\nMatrix Factorization CF, RMSE = %.4f" %RMSE)