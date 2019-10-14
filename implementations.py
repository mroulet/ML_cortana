# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np

def compute_loss(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse  
# ***************************************************
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    return (-1/y.shape[0])*(np.transpose(tx).dot(y-np.dot(tx,w)))
# ***************************************************    
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [] ## peut-être à enlever si pas nécessaire
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        # TODO: compute gradient and loss
        # ***************************************************
        
        # ***************************************************
        w = w-gamma*gradient
        # TODO: update w by gradient
        # ***************************************************
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
# ***************************************************
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return (-1/y.shape[0])*(tx.T.dot(y-tx.dot(w)))
# ***************************************************
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
# ***************************************************
def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_0,x_0 in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            sgd = compute_stoch_gradient(y_0,x_0,w)
            loss = compute_loss(y_0,x_0,w)
            # TODO: compute stoch gradient and loss
        
            w = w-gamma*sgd
            # TODO: update w by stoch gradient
        
            # store w and loss
            ws.append(w)
            losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
##******************************************************************
def least_squares_GD(y, tx):
    """calculate the least squares solution."""
    xT = np.transpose(tx)
    w = gradient_descent(y,tx,initial_w,max_iters,gamma)
    mse = compute_loss(y,tx,w)
    
    return mse, w
##******************************************************************
def least_squares_SGD(y, tx):
    """calculate the least squares solution."""
    xT = np.transpose(tx)
    w = gradient_descent(y,tx,initial_w,max_iters,gamma)
    mse = compute_loss(y,tx,w)
    
    return mse, w
##******************************************************************
def least_squares(y, tx):
    """calculate the least squares solution."""
    xT = np.transpose(tx)
    w = np.linalg.solve(xT.dot(tx),xT.dot(y)) # more efficient command
    mse = compute_loss(y,tx,w)
    
    return mse, w
##******************************************************************
"""
def ridge_regression(y, tx, lambda_):
    implement ridge regression.
    xT = np.transpose(tx)
    N = len(tx)
    D = len(xT)
    
    lambda_prime = lambda_*(2*N)
    a = (tx.T.dot(tx)+lambda_prime*np.identity(D))
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    
    mse = compute_loss(y,tx,w)
    
    #ridge = mse + lambda_*numpy.linalg.norm(w)
    
    return mse, w
"""
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)
##******************************************************************
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    pol_M = np.ones((len(x),degree+1))
    xs = 1
    for i in range(degree): # to be changed?
        xs = xs*x
        pol_M[:,i+1]=xs
    return pol_M
##******************************************************************
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
        
    N = len(y)
    indices = np.random.permutation(N)
    size = int(np.floor(ratio*N))
    train_index = indices[:size]
    test_index = indices[size:]

    x_tr = x[train_index]
    y_tr = y[train_index]
    x_te = x[test_index]
    y_te = y[test_index]
    
    return x_tr, x_te, y_tr, y_te
##******************************************************************
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
##******************************************************************
def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    # get k'th subgroup in test, others in train: TODO
    xk_subgroup = x[k_indices[k]]
    y_te = y[k_indices[k]]
    k_indices_remaining = np.delete(k_indices, k, 0)
    xk_remaining = np.ravel(x[k_indices_remaining])
    y_tr = np.ravel(y[k_indices_remaining])

    # form data with polynomial degree: TODO
    tx_te = build_poly(xk_subgroup, degree)
    tx_tr = build_poly(xk_remaining, degree)
    
    # ridge regression: TODO
    weight = ridge_regression(y_tr, tx_tr, lambda_)

    # calculate the loss for train and test data: TODO
    loss_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, weight))
    loss_te = np.sqrt(2 * compute_loss(y_te, tx_te, weight))

    return loss_tr, loss_te
##******************************************************************