import numpy as np
import torch
import scipy.sparse as sp



def get_labels(labels):
    classes = set(labels)
    classes_dict = {c: i for i,c in enumerate(classes)}
    labels = np.array([classes_dict[i] for i in labels])
    return labels
    

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    inputs = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    idx = np.array(inputs[:, 0], dtype=np.int32)
    features = np.array(inputs[:, 1:-1], dtype=np.float32)
    labels = get_labels(inputs[:,-1])

    "Building the graph"
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    A_tilde = np.eye(labels.shape[0])
    for v1,v2 in edges:
        A_tilde[v1][v2]=1
        A_tilde[v2][v1]=1

    adj=normalize(A_tilde)
    features = normalize(features)

    return torch.FloatTensor(adj), torch.tensor(features), torch.tensor(labels)


def normalize(mx):
    """Row-normalize matrix - Unchanged from Kipfs"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# adj, features, labels = load_data()
# print(adj.shape)
# print(features.shape)
# print(labels.shape)