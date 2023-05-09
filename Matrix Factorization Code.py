# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 20:20:55 2023

@author: colli
"""



import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import sparse
from scipy.sparse import csr_matrix

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    np.random.seed(941)
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break

    R2 = np.dot(P, np.transpose(Q.T))
    pos_in = R[np.nonzero(R)]
    pos_out = R2[np.nonzero(R)]

    MSE = np.mean(pow((pos_in - pos_out), 2))
    RMSE = np.sqrt(MSE)



    return P, Q.T, step, RMSE, R2, pos_in

###############################################################################


if __name__ == "__main__":
    columns_name = ['user_id', 'movie_id', 'rating', 'timestamp']
    data = pd.read_csv(
        "C:/Users/colli/OneDrive/Documents/movielens_100k.base",
        sep="\t", names=columns_name)

    users = np.arange(1, 943 + 1)
    movies = np.arange(1, 1682 + 1)
    shape = (len(users), len(movies))

    userc = CategoricalDtype(categories=sorted(users), ordered=True)
    moviec = CategoricalDtype(categories=sorted(movies), ordered=True)
    useri = data["user_id"].astype(userc).cat.codes
    moviei = data["movie_id"].astype(moviec).cat.codes

    coo = sparse.coo_matrix((data["rating"], (useri, moviei)), shape=shape)
    csr = coo.tocsr()
    base1 = csr.todense()
    R = np.array(base1)

    N = len(R)
    M = len(R[0])
    K = 2

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    nP, nQ , step, RMSE, R2, pos_in = matrix_factorization(R, P, Q, K)

print(nP)              
print(np.transpose(nQ)) 
print(RMSE)             
print(len(R2))
print(len(R2[0]))

print(len(pos_in))      



test = pd.read_csv("C:/Users/colli/OneDrive/Documents/movielens_100k.test",
                   sep="\t",names=columns_name)

useri_test = test["user_id"].astype(userc).cat.codes
moviei_test = test["movie_id"].astype(moviec).cat.codes

coo = sparse.coo_matrix((test["rating"], (useri_test, moviei_test)), shape=shape)
csr = coo.tocsr()
base1 = csr.todense()
base = np.array(base1)

pos_in = base[np.nonzero(base)]
pos_out = R2[np.nonzero(base)]

MSE = np.mean(pow((pos_in - pos_out), 2))
RMSE = np.sqrt(MSE)

print(RMSE)
print(len(pos_in))

