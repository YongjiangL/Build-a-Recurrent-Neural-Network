import numpy as np
from rnn_utils import *


"""
Implement a single forward step of LSTM-cell.
1. concatenate a<t-1> and x<t> in single matrx: concat = [a<t-1> x<t>]
2. compute all the formulas, use sigmoid and np.tanh()
forget gate: ft = sigmoid(Wf*concat + bf)
update gate: ut = sigmoid(Wu*concat + bf)
update cell: cct = tanh(Wc*concat + bc)
            c_next = ft*c_prev + ut*cct
output gate: ot = sigmoid(Wo*concat + bo)
a_next = ot*tanh(c_next)
3. compute the prediction y<t>, use softmax. 
"""
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Arguments:
    xt -- input data x<t>, numpy array of shape (n_x, m)
    a_prev -- hidden state at time 't-1', numpy array of shape (n_a, m)
    c_prev -- memory state at time 't-1', shape (n_a, m)
    parameters -- dictionary containing:
                Wf -- Weight matrix of forget gate, shape (n_a, n_a + n_x)
                bf -- bias of forget gate, shape (n_a, 1)
                Wu -- Weight matrix of update gate, shape (n_a, n_a + n_x)
                bu -- bias of update gate, shape (n_a, 1)
                Wc -- Weight matrix of first 'tanh', shape (n_a, n_a + n_x)
                bc -- bias of first 'tanh', shape (n_a, 1)
                Wo -- Weight matrix of output gate, shape (n_a, n_a + n_x)
                bo -- bias of output gate, shape (n_a, 1)
                Wy -- weight matrix to output, shape (n_y, n_a)
                by -- bias to output, shape (n_y, 1)
    Return:
            a_next -- next hidden state, shape (n_a, m)
            c_next -- next memory state, shape (n_a, m)
            yt_pred -- prediction of yt, shape (n_y, m)
            cache -- tuple of values for backward
    """
    # retrieve parameters
    Wf = parameters['Wf']
    bf = parameters['bf']
    Wu = parameters['Wu']
    bu = parameters['bu']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wy = parameters['Wy']
    by = parameters['by']

    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # concatenate
    concat = np.zeros((n_a + n_x, m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt

    # compute: ft/ut/ot stand for forget/update/output gates,
    # cct stands for candidate value (c tilde), c stands for memory
    ft = sigmoid(np.dot(Wf, concat) + bf)
    ut = sigmoid(np.dot(Wu, concat) + bf)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = ft*c_prev + ut*cct
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot*np.tanh(c_next)

    yt_pred = softmax(np.dot(Wy, a_next) + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, ut, cct, ot, xt, parameters)
    return a_next, c_next, yt_pred, cache






np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
c_prev = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wu = np.random.randn(5, 5+3)
bu = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", c_next.shape)
print("c_next[2] = ", c_next[2])
print("c_next.shape = ", c_next.shape)
print("yt[1] =", yt[1])
print("yt.shape = ", yt.shape)
print("cache[1][3] =", cache[1][3])
print("len(cache) = ", len(cache))


    


  
