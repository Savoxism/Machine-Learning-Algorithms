from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class UserUserCollaborativeFiltering(object):
    def __init__(self, Y_data, k, sim_func=cosine_similarity):
        self.Y_data = Y_data # a 2D array with shape (n_users, 3), each row is [user_id, item_id, rating]
        self.k = k 
        self.sim_func = sim_func
        self.Ybar = None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def fit(self):
        # Normalize the Y_data -> YBar
        users = self.Y_data[:, 0]
        self.Ybar = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))

        for n in range(self.n_users):
            # Row indices of ratings for user n
            ids = np.where(users == n)[0].astype(np.int32)
            # Indices of all items rated by user n
            item_ids = self.Y_data[ids, 1]
            # Ratings for user n
            ratings = self.Y_data[ids, 2]
            # Avoid 0 division
            self.mu[n] = np.mean(ratings) if ids.size > 0 else 0
            self.Ybar[ids, 2] = ratings - self.mu[n]

        # Form the rating matrix as a sparse matrix
        self.Ybar = sparse.coo_matrix((self.Ybar[:, 2],
                                       (self.Ybar[:, 1], self.Ybar[:, 0])),
                                       (self.n_items, self.n_users)).tocsr()
        self.similarity_matrix = self.sim_func(self.Ybar.T, self.Ybar.T)
        print(self.similarity_matrix.shape)

    def predict(self, u, i):
        """Predict the rating of user u for item i"""
        # Find item i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)

        # Find all users who have rated item i 
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)

        # Similarity of u and users who have rated i
        sim = self.similarity_matrix[u, users_rated_i]

        # Get the top k similar users
        nearest_neighbors = np.argsort(sim)[-self.k:]
        nearest_similarities = sim[nearest_neighbors]

        # Obtain the ratings of the nearest neighbors
        r = self.Ybar[i, users_rated_i[nearest_neighbors]].toarray().flatten()
        eps = 1e-8

        pred = (r*nearest_similarities).sum() / (np.abs(nearest_similarities).sum() + eps)

        unnormalized_pred = pred + self.mu[u]

        return unnormalized_pred
    

# Application
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv("ml-100k/ua.base", sep='\t', names=r_cols)
ratings_test = pd.read_csv("ml-100k/ua.test", sep='\t', names=r_cols)

# Training set
rate_train = ratings_base.values
rate_test = ratings_test.values

# Convert to 0-indexed
# rate_train[:, :2] -= 1
# rate_test[:, :2] -= 1

# rs = UserUserCollaborativeFiltering(rate_train, k=40)
# rs.fit()

# n_tests = rate_test.shape[0]
# SE = 0
# for n in range(n_tests):
#     pred = rs.predict(rate_test[n, 0], rate_test[n, 1])
#     SE += (pred - rate_test[n, 2]) ** 2

# RMSE = np.sqrt(SE / n_tests)
# print("User-User CF RMSE: ", RMSE)

# ================================================ #
# item-item collaborative filtering
rate_train = rate_train[:, [1, 0, 2]]
rate_test = rate_test[:, [1, 0, 2]]

rs = UserUserCollaborativeFiltering(rate_train, k=40)
rs.fit()

n_tests = rate_test.shape[0]
SE = 0
for n in range(n_tests):
    pred = rs.predict(rate_test[n, 0], rate_test[n, 1])
    SE += (pred - rate_test[n, 2]) ** 2

RMSE = np.sqrt(SE / n_tests)
print("Item-Item CF RMSE: ", RMSE)


    
