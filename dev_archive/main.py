import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

from sklearn.datasets import make_blobs, make_swiss_roll
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import k_means

from spektral.utils.convolution import normalized_adjacency
from spektral.layers.convolutional import GCSConv
from spektral.layers import MinCutPool
from sklearn.metrics.cluster import completeness_score, homogeneity_score, v_measure_score

from tensorflow.keras import Input, Model
import scipy.sparse as sp

from tsp_solver import tsp_solve
from spek_gnn_clustering import sp_matrix_to_sp_tensor


@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        _, S_pool = model(inputs, training=True)
        loss = sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return model.losses[0], model.losses[1], S_pool


num_KNN = 20  # K nearset neighbour parameter
n_channels = 16
import_data = True

if import_data:
    from cifar10loader import unpickle
    cifar_data = unpickle(os.path.join('../cifar-10-batches-py', 'data_batch_2'))

    X = cifar_data[b'data']
    y = cifar_data[b'labels']

    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    # plt.imshow(image_batch[5])

    image_batch = tf.convert_to_tensor(X)
    for i in range(10000):
        X[i] = tf.image.per_image_standardization(image_batch[i]).numpy()
        # normalise
    X = X.reshape((10000, -1))
    cluster_num = 10  # 聚类个数k

else:
    node_num = 500  # 节点个数
    n_x_dims = 3  # 节点特征维度
    cluster_num = 4  # 聚类个数k

    data = 'swiss roll'  # 'blob', 'swiss roll'
    data_random_state = 1

    ###########################################################################
    # Prepare the data
    ###########################################################################
    if data == 'blob':
        X, y = make_blobs(n_samples=node_num, centers=cluster_num, n_features=n_x_dims, random_state=data_random_state)
        # n_clust = y.max() + 1

    elif data == 'swiss roll':
        X, t = make_swiss_roll(n_samples=node_num, noise=0.2, random_state=data_random_state)

X = X.astype(np.float32)
A = kneighbors_graph(X, n_neighbors=num_KNN, mode='distance').todense()
# argument explanation: mode='distance', weighted adjacency matrix, mode=’connectivity’, binary adjacency matrix

A = np.asarray(A)
A = np.maximum(A, A.T)
A = sp.csr_matrix(A, dtype=np.float32)
A_norm = normalized_adjacency(A)
A_norm = sp_matrix_to_sp_tensor(A_norm)

############################################################################
# MODEL
############################################################################
np.random.seed(1)
epochs = 2000  # Training iterations
lr = 5e-4  # Learning rate

X_in = Input(shape=(X.shape[1],), name='X_in', dtype='float32')
A_in = Input(shape=(None, None), name='A_in', dtype='float32', sparse=True)

X_1 = GCSConv(channels=n_channels, activation='elu')([X_in, A_in])

pool1, adj1, s1 = MinCutPool(k=cluster_num,
                             mlp_hidden=None,
                             mlp_activation='relu',
                             return_mask=True)([X_1, A_in])

model = Model(inputs=[X_in, A_in], outputs=[pool1, s1])

################################################################################
# TRAINING
################################################################################
# Setup
inputs = [X, A_norm]
opt = tf.keras.optimizers.Adam(learning_rate=lr)
# Fit model
loss_history = []
nmi_history = []
for _ in tqdm(range(epochs)):
    outs = train_step(inputs)
    outs = [o.numpy() for o in outs]
    loss_history.append((outs[0], outs[1], (outs[0] + outs[1])))
    s_out = np.argmax(outs[2], axis=-1)
    nmi_history.append(v_measure_score(y, s_out))
loss_history = np.array(loss_history)
################################################################################
# RESULTS
################################################################################
_, s_out = model(inputs, training=False)
s_out = np.argmax(s_out, axis=-1)

# Labels analysis
hom = homogeneity_score(y, s_out)
com = completeness_score(y, s_out)
nmi = v_measure_score(y, s_out)
print("GNN clustering, Homogeneity: {:.3f}; Completeness: {:.3f}; NMI: {:.3f}".format(hom, com, nmi))

################################################################################
# Comparison with k-means
################################################################################

_, kMeans_out, _ = k_means(X, n_clusters=cluster_num)

# Labels analysis
hom = homogeneity_score(y, kMeans_out)
com = completeness_score(y, kMeans_out)
nmi = v_measure_score(y, kMeans_out)
print("k-means clustering, Homogeneity: {:.3f}; Completeness: {:.3f}; NMI: {:.3f}".format(hom, com, nmi))

# Plots
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(loss_history[:, 0], label="MinCUT loss")
plt.plot(loss_history[:, 1], label="Ortho. loss")
plt.plot(loss_history[:, 2], label="Total loss")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Iteration")

plt.subplot(122)
plt.plot(nmi_history, label="NMI")
plt.legend()
plt.ylabel("NMI")
plt.xlabel("Iteration")

################################################################################
# Visualisation of data
################################################################################

ThreeDimFig = plt.figure()
ax = ThreeDimFig.add_subplot(121, projection='3d')
kx = ThreeDimFig.add_subplot(122, projection='3d')

marker_list = ['1', '2', '3', '4', '5', '6']
colour_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

total_distance = []
total_distance_k = []

for i in range(cluster_num):
    indC = np.squeeze(np.argwhere(s_out == i))
    indC_k = np.squeeze(np.argwhere(kMeans_out == i))

    X_C = X[indC]
    X_C_k = X[indC_k]

    ax.scatter(X_C[:, 0], X_C[:, 1], X_C[:, 2], c='{}'.format(colour_list[i]), marker='${}$'.format(i))
    kx.scatter(X_C_k[:, 0], X_C_k[:, 1], X_C_k[:, 2], c='{}'.format(colour_list[i]), marker='${}$'.format(i))

    _, d = tsp_solve(X_C)
    total_distance.append(d)
    _, d_k = tsp_solve(X_C_k)
    total_distance_k.append(d_k)

    print('TSP solved for cluster {}'.format(i))

plt.show()
print('GNN overall distance: {}'.format(sum(total_distance)))
print('k-means overall distance: {}'.format(sum(total_distance_k)))
