import numpy as np
import scipy.sparse as sp
import torch
from scipy import sparse
import scipy.io as sio
from scipy.sparse import csc_matrix
from sklearn.metrics import f1_score
import random
import os
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def seed_everything(seed=616):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W

def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G = MyScaleSimMat(G)
    G = csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k
    return A

def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI < 0] = 0
    return PPMI

def normalize_sparse_hypergraph_symmetric(H) -> object:
    rowsum = np.array(H.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    D = sp.diags(r_inv_sqrt)

    colsum = np.array(H.sum(0))
    r_inv_sqrt = np.power(colsum, -1).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    B = sp.diags(r_inv_sqrt)

    Omega = sp.eye(B.shape[0])
    hx1 = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)

    return hx1

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')

    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def calculate_f1(pred,target):
    pred = np.argmax(pred, 1)
    f1_micro = f1_score(y_true=target, y_pred=pred, average='micro')
    f1_macro = f1_score(y_true=target, y_pred=pred, average='macro')
    return f1_micro, f1_macro

def load_source(source):
    smat = sio.loadmat('./dataset/' + source + '.mat')
    sx = smat['attribute']
    sy = smat['label']
    sy = sy.reshape(-1)
    sA = smat['net']
    s_A_k = AggTranProbMat(sA, 3)
    sPPMI = ComputePPMI(s_A_k)
    sx = normalize(sx)
    s_pos_weight = float(sPPMI.shape[0] * sPPMI.shape[0] - sPPMI.sum()) / sPPMI.sum()
    s_pos_weight = np.array(s_pos_weight).reshape(1, 1)
    s_pos_weight = torch.from_numpy(s_pos_weight)
    s_norm = sPPMI.shape[0] * sPPMI.shape[0] / float((sPPMI.shape[0] * sPPMI.shape[0] - sPPMI.sum()) * 2)
    sPPMI_ori = sPPMI + np.eye(sPPMI.shape[0])
    sPPMI_norm = normalize_adj(sparse.csr_matrix(sPPMI_ori))
    sPPMI_ori = torch.FloatTensor(sPPMI_ori)
    sPPMI_norm = sparse_mx_to_torch_sparse_tensor(sPPMI_norm)
    sx = torch.FloatTensor(sx.toarray())
    return sx, sy, sPPMI_ori, sPPMI_norm, s_norm, s_pos_weight

def load_target(target):
    tmat = sio.loadmat('./dataset/' + target + '.mat')
    tx = tmat['attribute']
    ty = tmat['label']
    ty = ty.reshape(-1)
    tA = tmat['net']
    train_list = tmat['train_idx']
    test_list = tmat['test_idx']
    t_A_k = AggTranProbMat(tA, 3)
    tPPMI = ComputePPMI(t_A_k)
    tx = normalize(tx)
    t_pos_weight = float(tPPMI.shape[0] * tPPMI.shape[0] - tPPMI.sum()) / tPPMI.sum()
    t_pos_weight = np.array(t_pos_weight).reshape(1, 1)
    t_pos_weight = torch.from_numpy(t_pos_weight)
    t_norm = tPPMI.shape[0] * tPPMI.shape[0] / float((tPPMI.shape[0] * tPPMI.shape[0] - tPPMI.sum()) * 2)
    tPPMI_ori = tPPMI + np.eye(tPPMI.shape[0])
    tPPMI_norm = normalize_adj(sparse.csr_matrix(tPPMI_ori))
    tPPMI_ori = torch.FloatTensor(tPPMI_ori)
    tPPMI_norm = sparse_mx_to_torch_sparse_tensor(tPPMI_norm)
    tx = torch.FloatTensor(tx.toarray())
    return tx, ty, tPPMI_ori, tPPMI_norm, t_norm, t_pos_weight, train_list, test_list


def load_H(sy, ty_train):
    H = np.zeros((sy.shape[0], ty_train.shape[0]))
    for i in range(sy.shape[0]):
        for j in range(ty_train.shape[0]):
            if sy[i] == ty_train[j]:
                H[i][j] = 1
    H_ori = H.copy()
    H_norm = normalize_sparse_hypergraph_symmetric(sparse.csr_matrix(H))
    Ht_norm = normalize_sparse_hypergraph_symmetric(sparse.csr_matrix(H.transpose()))
    H_norm = sparse_mx_to_torch_sparse_tensor(H_norm)
    Ht_norm = sparse_mx_to_torch_sparse_tensor(Ht_norm)
    H_ori = torch.FloatTensor(H_ori)

    return H_ori, H_norm, Ht_norm


