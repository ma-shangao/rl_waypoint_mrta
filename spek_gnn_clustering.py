import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import Input, Model
from tqdm import tqdm

from sklearn.neighbors import kneighbors_graph
from spektral.layers.convolutional import GCSConv
from spektral.layers import MinCutPool

from spektral.utils.convolution import normalized_adjacency
from sklearn.metrics.cluster import v_measure_score

# from rl_tsp_solver import rl_tsp_solver
# from tsp_solver import tsp_solve


def sp_matrix_to_sp_tensor(x):
    """
    Converts a Scipy sparse matrix to a tf.SparseTensor
    :param x: a Scipy sparse matrix
    :return: tf.SparseTensor
    """
    if not hasattr(x, 'tocoo'):
        try:
            x = sp.coo_matrix(x)
        except TypeError:
            raise TypeError('x must be convertible to scipy.coo_matrix')
    else:
        x = x.tocoo()
    return tf.SparseTensor(
        indices=np.array([x.row, x.col]).T,
        values=x.data,
        dense_shape=x.shape
    )


def _get_knn_adj_matrix(node_matrix, num_knn, knn_mode):
    adj_matrix = kneighbors_graph(node_matrix, n_neighbors=num_knn, mode=knn_mode).todense()
    # argument explanation: mode='distance', weighted adjacency matrix, mode=’connectivity’, binary adjacency matrix
    return adj_matrix


def _normalise_adj(adj_matrix):
    adj_matrix = np.asarray(adj_matrix)
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    adj_matrix = sp.csr_matrix(adj_matrix, dtype=np.float32)

    norm_adj = normalized_adjacency(adj_matrix)
    norm_adj = sp_matrix_to_sp_tensor(norm_adj)
    return norm_adj


class GnnClustering:
    def __init__(self, cluster_num, num_knn=5, knn_mode='distance', n_channels=16, tsp_feedback=False):
        self.s_out = None
        self.adj_matrix = None
        self.inputs = None
        self.nmi_history = None
        self.loss_history = None
        self.n_channels = n_channels
        self.cluster_num = cluster_num
        self.num_knn = num_knn
        self.knn_mode = knn_mode
        self.tsp_feedback = tsp_feedback

    def _model_setup(self, dim_feature):
        x_in = Input(shape=(dim_feature,), name='X_in', dtype='float32')
        adj_in = Input(shape=(None, None), name='A_in', dtype='float32', sparse=True)

        x_1 = GCSConv(channels=self.n_channels, activation='elu')([x_in, adj_in])

        pool1, adj1, s1 = MinCutPool(k=self.cluster_num,
                                     mlp_hidden=None,
                                     mlp_activation='relu',
                                     return_mask=True)([x_1, adj_in])

        self.model = Model(inputs=[x_in, adj_in], outputs=[pool1, s1])

    @tf.function
    def _train_step(self, inputs, opt):
        with tf.GradientTape() as tape:
            _, s_pool = self.model(inputs, training=True)
            loss = sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return self.model.losses[0], self.model.losses[1], s_pool

    def fit(self, x, label=None, epochs=2000, learning_rate=5e-4):
        self._model_setup(x.shape[1])
        # Setup
        self.adj_matrix = _get_knn_adj_matrix(x, self.num_knn, self.knn_mode)
        self.inputs = [x, self.adj_matrix]
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Fit model
        self.loss_history = []
        self.nmi_history = []
        for _ in tqdm(range(epochs)):
            outs = self._train_step(self.inputs, opt)
            outs = [o.numpy() for o in outs]
            self.loss_history.append((outs[0], outs[1], (outs[0] + outs[1])))
            s_out = np.argmax(outs[2], axis=-1)
            if label is not None:
                self.nmi_history.append(v_measure_score(label, s_out))
        self.loss_history = np.array(self.loss_history)

    def eval(self):
        _, s_out = self.model(self.inputs, training=False)
        self.s_out = np.argmax(s_out, axis=-1)
