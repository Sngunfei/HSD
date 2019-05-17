from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_embeddings(nodes, embeddings, label=False, labels=None, method="pca", node_name=None):

    if method == 'pca':
        model = PCA(n_components=2, whiten=True)
    else:
        model = TSNE(n_components=2,  random_state=42, n_iter=1000)

    node_pos = model.fit_transform(embeddings)

    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']

    if not label:
        for i in range(len(node_pos)):
            plt.scatter(node_pos[i, 0], node_pos[i, 1])
            plt.text(node_pos[i, 0], node_pos[i, 1], nodes[i])
        plt.show()
        return

    markers = ['<', '8', 's', '*', 'H', 'x', 'D', '>', '^', "v", '1', '2', '3', '4', 'X', '.']
    color_idx = {}
    for i in range(len(nodes)):
        color_idx.setdefault(labels[i], [])
        color_idx[labels[i]].append(i)

    for c, idx in color_idx.items():
        #plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, marker=markers[int(c)%16])#, s=area[idx])
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, marker=markers[int(c)%16])#, s=area[idx])
        #plt.text(node_pos[idx, 0], node_pos[idx, 1], idx)

    #for i in range(len(nodes)):
    #    plt.text(node_pos[i, 0], node_pos[i, 1], str(nodes[i]))

    plt.legend()
    plt.show()



def f(a):
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    w, v = np.linalg.eig(a)
    print(w)
    w = np.exp(-w)
    sort_indices = np.argsort(-w)
    w = -np.sort(-w)
    print(w)
    print(np.sum(w))
    w = np.diag(w)
    v = v[:, sort_indices]
    M = np.dot(v, np.dot(w, np.linalg.inv(v)))
    print(M)
    z = M
    """
    for i in range(1000):
        z = np.dot(z, M)
        print(z)
    """
    # 特征值0对应的特征向量是[1,1,1,1,...]，


def laplacian_norm(a):
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    I = np.diag(np.array([1] * len(a)))
    T = np.diag(np.diag(a) ** 0.5)
    T_inv = np.diag(np.diag(a) ** -0.5)
    L = np.dot(T_inv, np.dot(a, T_inv))
    P = np.dot(T_inv, np.dot(I - L, T))
    print(P)
    w, v = np.linalg.eig(L)
    w = -np.sort(-w)
    order = -np.argsort(-w)
    v = v[:, order]
    print(w)
    print(v)
    print(a)
    print(L)
    w = np.diag(np.exp(-w))
    print(w)
    M = np.dot(v, np.dot(w, np.linalg.inv(v)))
    print(np.sum(np.abs(M), 1))


def fff():
    a = np.array([5,5,5,5,5])
    b = np.array([4,6,4,6])
    print(a-b)

if __name__ == '__main__':
    fff()
    pass
    a = np.array([[2, -1, -1, 0, 0, 0, 0, 0],
                  [-1, 3, 0, -1, -1, 0, 0, 0],
                  [-1, 0, 2, 0, 0, -1, 0, 0],
                  [0, -1, 0, 2, 0, 0, -1, 0],
                  [0, -1, 0, 0, 1, 0, 0, 0],
                  [0, 0, -1, 0, 0, 1, 0, 0],
                  [0, 0, 0, -1, 0, 0, 2, -1],
                  [0, 0, 0, 0, 0, 0, -1, 1]])
    b = np.array([[2, -1, -1, 0],
                  [-1, 2, 0, -1],
                  [-1, 0, 1, 0],
                  [0, -1, 0, 1]])
    #f(a)
    #laplacian_norm(a)