# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np

## *************************************************** 
## *************************************************** 
## GRADIENT DESCENT
## *************************************************** 
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''CORRECT verifié avec data exo 2'''
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
    # Define parameters to store w and loss
    threshold = 1e-8
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
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]
## ***************************************************
## ***************************************************
## LEAST SQUARE SGD
## ***************************************************
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
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
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_0,x_0 in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            
            # compute stoch gradient and loss
            sgd = compute_gradient(y_0,x_0,w)
            loss = compute_loss(y_0,x_0,w)
            
            # update w by stoch gradient
            w = w-gamma*sgd
            
            # store w and loss
            ws.append(w)
            losses.append(loss)
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
            
    return ws[-1], losses[-1]
##******************************************************************
##******************************************************************
## LEAST SQUARE USING NORMAL EQUATIONS
##******************************************************************
def least_squares(y, tx):
    """ Least squares regression using normal equations
    Arguments:
        y: labels
        tx: features
    Returns:
        w: optimized weight vector
        loss: optimized mean squared error
    """
    xT = tx.T
    w = np.linalg.solve(xT.dot(tx),xT.dot(y)) # more efficient command
    mse = compute_loss(y,tx,w)
    
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
## HELPERS
## ******************************************************************
def compute_loss(y, tx, w):
    """compute the loss by mse."""
    # error vector
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    
    return mse  
# ***************************************************
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # error vector
    e = y - tx.dot(w)
    
    return - (tx.T.dot(e)) / e.size
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
            
##******************************************************************

def build_poly(tx, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    poly_tx = np.ones((len(tx), 1))
    # iterate through features column
    columns = tx.shape[1] if len(tx.shape) > 1 else 1
    
    for col in range(columns):

        # build polynomial
        for degree in range(1, degree + 1):
            
            if columns >1:
                poly_tx = np.c_[poly_tx, np.power(tx[ :, col], degree)]
            else:
                poly_tx = np.c_[poly_tx,np.power(x,degree)]
        
        # once i feature poly built, add vector of ones for i+1 feature
        if tx.shape[1] > 1 and col != tx.shape[1] - 1:
            poly_tx = np.c_[poly_tx, np.ones((len(tx), 1))] 
     
    return poly_tx

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
    
    # put k'th subgroup in test, others in train
    y_test = y[k_indices[k]]
    xk_test = x[k_indices[k]]
    
    y_train = np.delete(y,k_indices[k])
    xk_train = np.delete(x,k_indices[k], axis = 0)
    
    # ridge regression
    weight, loss = ridge_regression(y_train, xk_train, lambda_)

    # calculate the loss for train and test data
    loss_train = np.sqrt(2 * compute_loss(y_train, xk_train, weight))
    loss_test = np.sqrt(2 * compute_loss(y_test, xk_test, weight))

    return loss_train, loss_test

##******************************************************************
##******************************************************************
## BROUILLON
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
        loss, w = logistic_regression(y, tx, w, max_iter, gamma)
        
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
##******************************************************************
def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # return loss, gradient, and hessian: TODO
    loss = calculate_loss(y, tx, w) +lambda_*np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2* lambda_*w
    
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


##******************************************************************
##******************************************************************
## PRE-PROCESSING OF DATA
##******************************************************************
##******************************************************************
def process_data(tx):
    
    # In each columns, we remove nan values and replace outliers by bound
    for i in range(tx.shape[1]):
        col = np.array(tx[:, i])
        # Replace -999 value by mean
        tx[:, i] = replace_data_point_by_mean(col)
        # replace outliers by lower/higher bound
        tx[:, i] = replace_outliers_by_bound(col)
    
    # remove features with no variance (mettre un treshold?)
    tx = remove_columns_with_no_variance(tx)
    
    return tx
##******************************************************************
def remove_columns_with_no_variance(tx):
    nb_cols = tx.shape[1]
    col_index = []


    for col in range(nb_cols):
        # If the columns have a unique element (all data have same value)
        if len(np.unique(tx[:,col])) == 1:
            col_index.append(col)
    
    # We remove the columns from our dataset
    nb_cols_deleted = 0
    for col in col_index:
        tx = np.delete(tx, col - nb_cols_deleted, 1)
        nb_cols_deleted += 1
    
    return tx
##******************************************************************
def  replace_data_point_by_mean(col):
    ''' replace missing datapoint of matrix x by the mean of each column
    '''
    # If there is a nan value in the feature column
    if -999 in col:
        ind_nan = []
        compressed =[]
        for i in range(len(col)):
            if col[i] == -999:
                ind_nan.append(i)
            else:
                compressed.append(col[i])
        mean = np.mean(compressed)
        
        for i in range(len(col)):
            if col[i] == -999:
                col[i] = mean
                
    return col
##******************************************************************
def replace_outliers_by_bound(col):
# Handling the outliers

    # compute mean and std of the column
    std = np.std(col)
    mean = np.mean(col)
        
    low_bound = mean - 2 * std
    up_bound = mean + 2 * std
        
    col[col < low_bound] = low_bound
    col[col > up_bound] = up_bound
    
    return col
##******************************************************************
def standardize(x, mean_x = None, std_x = None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x
##******************************************************************
def test_score(y, tx, w, verbose = True):
    """
    Reports the percentage of correct predictions of a model that is applied
    on a set of labels.
    Args:
        y: numpy array of labels for testing purpose
        tx: numopy array of features in the learned data set
        w_best: the optimized weight vector of the model
    Returns:
        correct_percentage: the percentage of correct predictions of the model 
            when it is applied on the given test set of labels
    """
    y_pred = tx.dot(w)
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1
    score = np.sum(y_pred == y) / float(len(y_pred))
    if verbose:
        print('Percentage of correct predictions is: %', score * 100)
    return score
##******************************************************************
def load_data(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metrics system."""
    path_dataset = "../data/height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50]
        weight = weight[::50]

    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])

    return height, weight, gender
##******************************************************************
def standardize(x, mean_x = None, std_x = None):
    """
    Standardizes the original data set.
    Args:
        x: data set to standardize
        mean_x: mean of the data set, can be specified or computed
        std_x: standard deviation of the data set, can be specified or computed
    Returns:
        x: standardized data set
        mean_x: mean of the data set
        std_x: standard deviation of the data set
    """
    if mean_x is None:
        mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x
##******************************************************************
def find_optimal_hyperparameters(
    y_train, processed_tx_train, y_test, processed_tx_test, degrees, lambdas, k_fold):
    
    min_loss_test = 10e6
    seed = 1
    
    k_indices = build_k_indices(y_train, k_fold, seed)

    for index_degree, degree in enumerate(degrees):
        print('Iteration over degrees:', degree)
        # Build polynomial tx
        poly_tx_train = build_poly(processed_tx_train, degree)
        
            
        for ind, lambda_ in enumerate(lambdas):
            
            # losses of each fold
            fold_losses_test = []
                
            for k in range(k_fold):
                _, loss_test = cross_validation(y_train, poly_tx_train, k_indices, k, lambda_, degree)
                
                # Store loss of fold k
                fold_losses_test.append(loss_test)
                
            # compute mean loss over all folds
            mean_loss_test = np.mean(fold_losses_test)
            
            # Optimal parameters condition
            if mean_loss_test < min_loss_test:
                print('\nBetter hyperparameters found:')
                print('- mean loss = ', mean_loss_test)
                min_loss_test = mean_loss_test 
                optimal_degree = degree
                optimal_lambda = lambda_
                print('- polynomial degree = ', optimal_degree)
                print('- lambda = ', optimal_lambda)
    
    return optimal_degree, optimal_lambda
##******************************************************************
def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx
##******************************************************************

