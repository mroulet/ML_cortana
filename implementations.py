# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np

# ***************************************************
## Pre processing of data
# ***************************************************
def  replace_data_point_by_mean(x):
    ''' replace missing datapoint of matrix x by the mean of each column
    '''
    
    col = x.shape[1]
    row = x.shape[0]
    
    for i in range(col):
        if -999 in x[:,i]:
            ind_nan = []
            compressed =[]
            for j in range (row):
                if x[j,i] == -999:
                    ind_nan.append(j)
                else:
                    compressed.append(x[j,i])
            mean = np.mean(compressed)
            
            x[:,i] = np.where(x[:,i] == -999, mean, x[:,i])
            
    return x
# ***************************************************
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
        # compute loss and gradient
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        # update w by gradient
        w = w-gamma*gradient        
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

    # split the data based on the given ratio        
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
def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    # compute the cost: TODO
    loss = calculate_loss(y, tx, w)
    # compute the gradient: TODO
    gradient = calculate_gradient(y, tx, w)
    # update w: TODO
    w = w-gamma*gradient

    return loss, w
##******************************************************************
def logistic_regression(y, tx, initial_w,max_iters, gamma):
    
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = logistic regression(y, tx, w, max_iter, gamma)
        
        # log info
        #if iter % 100 == 0:
        #    print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        #    print(w)
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, loss
##******************************************************************
def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
    
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        #if iter % 100 == 0:
        #    print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, loss
##******************************************************************
def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # calculate hessian: TODO
    ones = np.ones(len(tx))
    Snn = sigmoid(tx.dot(w))*(ones-sigmoid(tx.dot(w)))
    diag_m = np.diag(np.diag(Snn))
    
    return tx.T.dot(diag_m).dot(tx)
##******************************************************************
def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    # return loss, gradient, and hessian
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hess = calculate_hessian(y, tx, w)
    
    return loss, gradient, hess
##******************************************************************
def learning_by_newton_method(y, tx, w):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    # return loss, gradient and hessian: TODO
    loss, gradient, hess = logistic_regression(y,tx,w)
    # update w: TODO
    w_new = w - np.linalg.inv(hess).dot(gradient)
    
    return loss, w_new
##******************************************************************
def calculate_loss(y, tx, w,):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1+np.exp(tx.dot(w))))-y.T.dot(tx.dot(w))
##******************************************************************
def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t) / (1+np.exp(t))
##******************************************************************
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(sigmoid(tx.dot(w))-y)
##******************************************************************
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    
    return x, mean_x, std_x
##******************************************************************
def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # return loss, gradient, and hessian: TODO
    loss = calculate_loss(y, tx, w) + lambda_*(np.sum(w**2))#/tx.shape[0]
    gradient = calculate_gradient(y, tx, w)
    
    return loss, gradient
##******************************************************************
def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # return loss, gradient: TODO
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    # update w: TODO
    w = w-gamma*gradient
    
    return loss, w