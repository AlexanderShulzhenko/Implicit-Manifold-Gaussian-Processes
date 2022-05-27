import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle, gpflow, os.path, os
import tensorflow as tf
import math
from sklearn.model_selection import train_test_split
from tqdm.notebook import trange
from graph_matern.kernels.graph_matern_kernel import GraphMaternKernel
from graph_matern.inducing_variables import GPInducingVariables
from graph_matern.svgp import GraphSVGP
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

dtype=tf.float64

gpflow.config.set_default_float(dtype)
gpflow.config.set_default_summary_fmt("notebook")
tf.get_logger().setLevel('ERROR')


class GraphGPC():
    def __init__(self,k_neig=40,epsilon=0.2,num_eigenpairs=1000,kappa=5,nu=3/2,sigma_f=1):
        self.k_neig=k_neig
        self.epsilon = epsilon
        self.num_eigenpairs=num_eigenpairs
        self.kappa=kappa
        self.nu=nu
        self.sigma_f=sigma_f
        self.model=None


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

    def fit(self, X, train_indices, ys_train, cls_number, train_num=1500):
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
        
        ''' Types '''
        nodes_train = nodes_train.reshape(-1,1)
        nodes_train = tf.convert_to_tensor(nodes_train, dtype=dtype)
        
        ys_train = ys_train.reshape(-1,1).astype('float')

        ''' Make Model'''
        
        inducing_points = GPInducingVariables(nodes_train)
        kernel = GraphMaternKernel((eigenvectors, eigenvalues), nu=self.nu, kappa=self.kappa, sigma_f=self.sigma_f, vertex_dim=0, point_kernel=None, dtype=dtype)

        data_train = (nodes_train,ys_train)
        #train_num = 1500#len(data_train)
        
        def opt_step(opt, loss, variables):
            opt.minimize(loss, var_list=variables)

        def optimize_SVGP(model, optimizers, steps, q_diag=True):
            if not q_diag:
                gpflow.set_trainable(model.q_mu, False)
                gpflow.set_trainable(model.q_sqrt, False)

            adam_opt, natgrad_opt = optimizers

            variational_params = [(model.q_mu, model.q_sqrt)]

            autotune = tf.data.experimental.AUTOTUNE
            data_minibatch = (
                tf.data.Dataset.from_tensor_slices(data_train)
                    .prefetch(autotune)
                    .repeat()
                    .shuffle(train_num)
                    .batch(train_num)
            )
            data_minibatch_it = iter(data_minibatch)
            loss = model.training_loss_closure(data_minibatch_it)
            adam_params = model.trainable_variables
            natgrad_params = variational_params

            adam_opt.minimize(loss, var_list=adam_params)
            if not q_diag:
                natgrad_opt.minimize(loss, var_list=natgrad_params)
            t = trange(steps)
            for step in t:
                opt_step(adam_opt, loss, adam_params)
                if not q_diag:
                    opt_step(natgrad_opt, loss, natgrad_params)
                if step % 50 == 0:
                    likelihood = model.elbo(data_train)
                    t.set_postfix({'ELBO': likelihood.numpy()})

        ''' Build model'''
        self.model = gpflow.models.SVGP(
                                kernel=kernel,
                                likelihood=gpflow.likelihoods.MultiClass(cls_number),
                                inducing_variable=inducing_points,
                                num_latent_gps=cls_number,
                                whiten=True,
                                q_diag=True,
                            )

        adam_opt = tf.optimizers.Adam(0.01)
        natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.01)

        optimize_SVGP(self.model, (adam_opt, natgrad_opt), 1000, True)

    def predict(self, test_indices, full_cov=True):
        
        ''' Types '''
        nodes_test = test_indices
        nodes_test = nodes_test.reshape(-1,1)
        nodes_test = tf.convert_to_tensor(nodes_test, dtype=dtype)
        
        ''' Make predictions'''
        y_pred_mean, y_pred_var = self.model.predict_y(nodes_test)
        y_pred = np.argmax(y_pred_mean.numpy(), axis=1).ravel()

        return y_pred, y_pred_var
