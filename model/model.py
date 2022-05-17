#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gpflow
import math
import scipy
from scipy.special import sph_harm, lpmv, factorial
from scipy.special import gamma
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from graph_matern.kernels.graph_matern_kernel import GraphMaternKernel

import networkx as nx
from scipy import sparse
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm.notebook import trange

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dtype = tf.float64


# In[219]:


class GraphGP():
    def __init__(self,k_neig=5,epsilon=1,num_eigenpairs=1000,kappa=4,nu=3,sigma_f=1):
        self.k_neig=k_neig
        self.epsilon = epsilon
        self.num_eigenpairs=num_eigenpairs
        self.kappa=kappa
        self.nu = nu
        self.sigma_f = sigma_f
        
    def optimize_GPR(self,model, train_steps):
        loss = model.training_loss
        trainable_variables = model.trainable_variables

        adam_opt = tf.optimizers.Adam(learning_rate=0.01)
        adam_opt.minimize(loss=loss, var_list=trainable_variables)

        t = trange(train_steps - 1)
        for step in t:
            self.opt_step(adam_opt, loss, trainable_variables)
            if step % 50 == 0:
                t.set_postfix({'likelihood': -model.training_loss().numpy()})

    @tf.function
    def opt_step(self,opt, loss, variables):
        opt.minimize(loss, var_list=variables)
        
    def build_graph(self,X,k_neig,eps):
        G = nx.Graph()
        neigh = KNeighborsClassifier(n_neighbors=k_neig)
        neigh.fit(X, np.zeros(len(X)))
        for i in range(len(X)):
            neighb_data = neigh.kneighbors(X[i].reshape(1,-1))
            for j in range(len(neighb_data[1][0])):
                d = neighb_data[0][0][j]
                G.add_edge(i,neighb_data[1][0][j], weight = np.exp( -d**2/(4*eps**2) ))
        return G

    def fit_and_predict(self, X, train_indices, ys_train, full_cov=False):
        ''' Build Graph'''
        
        G = self.build_graph(X,self.k_neig,self.epsilon)
        nodes = np.array(G.nodes)

        num_eigenpairs = self.num_eigenpairs

        laplacian = sparse.csr_matrix(nx.laplacian_matrix(G,nodelist=np.sort(nodes)), dtype=np.float64)
        if num_eigenpairs >= len(G):
            print("Number of features is greater than number of vertices. Number of features will be reduced.")
            num_eigenpairs = len(G)

        eigenvalues, eigenvectors = tf.linalg.eigh(laplacian.toarray())
        eigenvectors, eigenvalues = eigenvectors[:, :num_eigenpairs], eigenvalues[:num_eigenpairs]
               
        eigenvalues, eigenvectors = tf.convert_to_tensor(eigenvalues, dtype=dtype), tf.convert_to_tensor(eigenvectors, dtype)
        
        ''' Make New ys and train-test split'''
        
        nodes_train = train_indices
        nodes_test = np.array(list(set(np.arange(len(X))) - set(nodes_train))) #sorted asc
        
        ''' Types '''
        nodes_train = nodes_train.reshape(-1,1)
        nodes_test = nodes_test.reshape(-1,1)
        nodes_train = tf.convert_to_tensor(nodes_train, dtype=dtype)
        nodes_test = tf.convert_to_tensor(nodes_test, dtype=dtype)
            
        ''' Make Model'''
        N = len(G)
        kernel = GraphMaternKernel((eigenvectors, eigenvalues), 
                                  nu=self.nu, 
                                  kappa=self.kappa, 
                                  sigma_f=self.sigma_f, 
                                  vertex_dim=0, 
                                  point_kernel=None, 
                                  dtype=dtype)
        model = gpflow.models.GPR(data=(nodes_train, ys_train.reshape(-1,1)), kernel=kernel, noise_variance=0.01)
        self.optimize_GPR(model, 1000)
        
        ''' Make Prediction'''
        mean, cov = model.predict_f(nodes_test, full_cov=full_cov)
        
        return mean.numpy().ravel(), cov

