# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """

    return 1/(2*len(y))*np.sum(((y-tx.dot(w))**2))    
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
    # ***************************************************
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
##******************************************************************
def least_squares_GD(y, tx):
    """calculate the least squares solution."""
    xT = np.transpose(tx)
    w = gradient descent(y,tx,initial_w,max_iters,gamma)
    mse = compute_loss(y,tx,w)
    
    return mse, w
##******************************************************************
def least_squares_SGD(y, tx):
    """calculate the least squares solution."""
    xT = np.transpose(tx)
    w = gradient descent(y,tx,initial_w,max_iters,gamma)
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
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(tx)
    D = len(xT)
    
    lambda_prime = lambda_*(2*N)
    a = (tx.T.dot(tx)+lambda_prime*np.identity(D))
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    
    mse = compute_loss(y,tx,w)
    
    #ridge = mse + lambda_*numpy.linalg.norm(w)
    
    return mse, w