# -*- coding: utf-8 -*-
"""Basic implementation functions derived from lab sessions."""
import numpy as np
from helpers import *


## *************************************************** 
## GRADIENT DESCENT
## *************************************************** 
def least_squares_GD(y, tX, w_initial, max_iters, gamma):
    
    """Linear regression using gradient descent algorithm
    Arguments:
        y: labels
        tx: features matrix
        initial_w: initial weight vector
        max_iters: maximum number of iteration
        gamma: step-size
    Returns:
        w: optimized weight vector
        loss: optimized mean squared error
    """
    
    # init parameters
    threshold = 1e-8
    losses = []
    w = w_initial
    
    for iter in range(max_iters):
       
        gradient = compute_gradient(y,tX,w)    # compute loss and gradient
        loss = compute_loss(y,tX,w)            
        
        w = w - gamma*gradient                   # update w by gradient

        losses.append(loss)
        
        # convergence criterion met
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break 
            
    return w, losses[-1]
## ***************************************************



## ***************************************************
## LEAST SQUARE SGD
## ***************************************************
def least_squares_SGD(y, tX, initial_w, batch_size, max_iters, gamma):
    """ Linear regression using stochastic gradient descent
    Arguments:
        y: labels
        tx: features matrix
        initial_w: initial weight vector
        max_iters: maximum number of iteration
        gamma: step-size
    Returns:
        w: optimized weight vector
        loss: optimized mean squared error
    """
    threshold = 1e-8
    losses = []
    w = initial_w
    
    for iter in range(max_iters):
        for y_0,x_0 in batch_iter(y, tX, batch_size, num_batches=1, shuffle=True):
            
            # compute stoch gradient and loss
            sgd = compute_gradient(y_0,x_0,w)
            loss = compute_loss(y_0,x_0,w)
            
            # update w by stoch gradient
            w = w - gamma*sgd

            losses.append(loss)
        
        # convergence criterion met
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break 
            
    return w, losses[-1]
##******************************************************************



##******************************************************************
## LEAST SQUARE USING NORMAL EQUATIONS
##******************************************************************
def least_squares(y, tX):
    """ Least squares regression using normal equations
    Arguments:
        y: labels
        tx: features
    Returns:
        w: optimized weight vector
        loss: optimized mean squared error
    """
    xT = tX.T
    w = np.linalg.solve(xT.dot(tX),xT.dot(y))
    mse = compute_loss(y,tX,w)
    
    return w, mse

## ******************************************************************




## ******************************************************************
## RIDGE REGRESSION
## ******************************************************************
def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations
    Arguments:
        y: labels
        tx: features
        lambda_: 
    Returns:
        w: optimized weight vector
        loss: optimized mean squared error
    """
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_loss(y,tx,w)
    
    return w, loss

## ******************************************************************




## ******************************************************************
## LOGISTIC REGRESSION
## ******************************************************************
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    '''Logistic regression using gradient descent or SGD
    Arguments:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: max iteration to learn
        gamma: step-size
    Returns:
        w: optimal weight vector that minimize the logistic loss
        loss: logistic loss
    '''
    
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = compute_logistic_loss(y,tx,w)

        gradient = compute_logistic_gradient(y,tx,w)
        w = w - gamma*gradient

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, loss

## ******************************************************************



## ******************************************************************
## PENALIZED LOGISTIC REGRESSION
## ******************************************************************
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma): 
    
    '''Regularized logistic regression using gradient descent or SGD
    Arguments:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: max iteration to learn
        gamma: step-size
    Returns:
        w: optimal weight vector that minimize the logistic loss
        loss: logistic loss
    '''

    threshold = 1e-8
    losses = []
    w = initial_w
    
    for iter in range(max_iters):
        
        # compute loss and gradient and update weights
        loss = compute_logistic_loss(y, tx, w) + lambda_*w.T.dot(w)
        
        gradient = compute_logistic_gradient(y, tx, w) + 2* lambda_*w
        w = w - gamma*gradient
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, loss
