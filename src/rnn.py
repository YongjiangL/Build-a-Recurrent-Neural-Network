import numpy as np
from rnn_utils import *

"""
Implement a single forward step of RNN-cell. We have m examples.
1. compute hidden state with tanh activation: a<t> = tanh(Waa*a<t-1> + Wax*x<t> + ba)
2. compute the output: y'<t> = softmax(Wya*a<t> + by)
3. store (a<t>, a<t-1>, x<t>, parameters) in cache for backward propagation
4. return (a<t>, y'<t>) and cache
"""
def rnn_cell_forward(xt, a_prev, parameters):
    """
    Arguments:
    xt -- input data x<t>, numpy array of shape (n_x, m)
    a_prev -- hidden state at time 't-1', numpy array of shape (n_a, m)
    parameters -- dictionary containing:
                Wax -- Weight matrix of xt, shape (n_a, n_x)
                Waa -- Weight matrix of a<t>, shape (n_a, n_a)
                Wya -- Weight matrix to output, shape (n_y, n_a)
                ba -- bias, shape (n_a, 1)
                by --- bias to output, shape (n_y, 1)
    Return:
            a_next -- next hidden state, shape (n_a, m)
            yt_pred -- prediction of yt, shape (n_y, m)
            cache -- tuple of values for backward
    """
    # Retrieve parameters
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    # compute next at
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    # compute output
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    # store values
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


def rnn_cell_backward(da_next, cache):
    """
    implement the backward pas for RNN-cell (one time-step)
    Arguments:
        da_next --- gradient of loss w.r.t. next hidden state
        cache -- output of rnn_cell_forward
    Returns:
        gradients -- dictionary containing:
                dx -- gradient of input, shape (n_x, m)
                da_prev -- gradient of previous state, shape (n_a, m)
                dWax -- gradient of input-hidden weight, shape (n_a, n_x)
                dWaa -- gradient of hidden-hidden weight, shape (n_a, n_a)
                dba -- gradient of bias, shape (n_a, 1)
    """
    # retrieve values
    (a_next, a_prev, xt, parameters) = cache

    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    # compute gradients
    dtanh = (1 - a_next ** 2)*da_next
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)
    dba = np.sum(dtanh, axis=1, keepdims=1)

    # store
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients

    
    


"""
Implement the forward propagation of RNN.
1. create a vector of zeros (a) that stores all hidden states computed by RNN
2. initialize initial hidden state a0
3. start looping over time step t:
    . update next hidden state and the cache by running rnn_cell_forward
    . store hidden state at time t in a
    . store prediction in y
    . add cache to list of caches
4. return a, y and caches
"""
def rnn_forward(x, a0, parameters):
    """
    Arguments:
    x -- input data, shape (n_x, m, T_x)
    a0 -- initial hidden state, shape (n_a, m)
    parameters -- dictionary containing:
                Wax -- Weight matrix of xt, shape (n_a, n_x)
                Waa -- Weight matrix of a<t>, shape (n_a, n_a)
                Wya -- Weight matrix to output, shape (n_y, n_a)
                ba -- bias, shape (n_a, 1)
                by --- bias to output, shape (n_y, 1)
    Return:
            a -- hidden state of each time, shape (n_a, m, T_x)
            y_pred -- prediction of each time, shape (n_y, m, T_x)
            cache -- tuple of values for backward
    """
    caches = []

    # retrieve dimensions
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape

    # initialize 'a' and 'y'
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    a_next = a0

    # loop over all times
    for t in range(T_x):
        # update next hidden state
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        # save
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)

    caches = (caches, x)
    return a, y_pred, caches



def rnn_backward(da, caches):
    """
    implement backward pass for RNN over an entire sequence of input data
    Arguments:
        da -- upstream gradients of all hiddent states, shape (n_a, m, T_x)
        caches -- from rnn_forward
    Returns:
        gradients -- dictionary
            dx -- gradient of input, shape (n_x, m, T_x)
            da0 -- gradient of initial hiddent state, shape (n_a, m)
            dWax -- gradient of input-hidden weight, shape (n_a, n_x)
            dWaa -- gradient of hidden-hidden weight, shape (n_a, n_a)
            dba -- gradient of bias, shape (n_a, 1)
    """
    # retrieve values
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0] # format: (a_next, a_prev, xt, parameters)

    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))

    # loop over all times
    for t in reversed(range(T_x)):
        # compute gradients
        gradients = rnn_cell_backward(da[:,:,t]+da_prevt, caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        dx[:,:,t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    da0 = da_prevt
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    
    return gradients

np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", a_next.shape)
print("yt_pred[1] =", yt_pred[1])
print("yt_pred.shape = ", yt_pred.shape)


np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
a, y_pred, caches = rnn_forward(x, a0, parameters)
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)
print("y_pred[1][3] =", y_pred[1][3])
print("y_pred.shape = ", y_pred.shape)
print("caches[1][1][3] =", caches[1][1][3])
print("len(caches) = ", len(caches))



np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
b = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)
da_next = np.random.randn(5,10)
gradients = rnn_cell_backward(da_next, cache)
print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients["dba"][4])
print("gradients[\"dba\"].shape =", gradients["dba"].shape)


np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
a, y, caches = rnn_forward(x, a0, parameters)
da = np.random.randn(5, 10, 4)
gradients = rnn_backward(da, caches)
print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients["dba"][4])
print("gradients[\"dba\"].shape =", gradients["dba"].shape)
