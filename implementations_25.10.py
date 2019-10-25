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
    for iter in range(max_iters):
        # compute loss and gradient
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        # update w by gradient
        w = w-gamma*gradient 
        
        '''
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        '''
        
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
    
    for iter in range(max_iters):
        for y_0,x_0 in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            
            # compute stoch gradient and loss
            sgd = compute_gradient(y_0,x_0,w)
            loss = compute_loss(y_0,x_0,w)
            
            # update w by stoch gradient
            w = w-gamma*sgd
            
            '''
            if iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            '''
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
        print(loss)
        '''
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        '''
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
    # return loss, gradient

    threshold = 1e-8
    losses = []
    w = initial_w
    for iter in range(max_iters):
        
        # compute loss and gradient and update weights
        loss = compute_logistic_loss(y, tx, w) +lambda_*w.T.dot(w)
        gradient = compute_logistic_gradient(y, tx, w) + 2* lambda_*w
        w = w-gamma*gradient
        
        # log info
        '''
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        '''
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, loss

## ******************************************************************
## ******************************************************************
## HELPERS
## ******************************************************************
def compute_logistic_loss(y, tx, w,):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1.+np.exp(tx.dot(w)))-y*tx.dot(w))

    '''
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)
    '''

## ******************************************************************
def sigmoid(t):
    """apply sigmoid function on t."""
    return 1. / (1+np.exp(-t))
## ******************************************************************
def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(sigmoid(tx.dot(w))-y)
## ****************************************************************** 
def convert_to_proba(y):
    y[y == -1] = 0
    return y
## ****************************************************************** 
def compute_loss(y, tx, w):
    """compute the loss by mse."""
    # error vector
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    
    return mse  
## ****************************************************************** 
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
    """Buil polynomial data from initial data. 
    Arguments:
        tx: features
        degree: degree of the polynomial
    """
    
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
def build_cross_terms(tx):
    '''
    Build the cross term array 
    Arguments:
        tx: features
    Returns:
        cross_terms: cross term array

    '''
    cross_terms = []
    
    for feature1 in range(tx.shape[1]):
        for feature2 in range(feature1 + 1, tx.shape[1],1):
            
            if feature1==0 and feature2 == 1:
                cross_terms = np.array(tx[:, feature1] * tx[:, feature2])
                
            else:
                cross_terms = np.c_[cross_terms, tx[:, feature1] * tx[:, feature2]]
                
    return cross_terms
    
##******************************************************************
def split_data(tx, y, ratio, seed=1):
    """
    Split data in to subdatasets according to ratio. If ratio is 0.8
    80% of the data will be used for learning and assigned to training. 
    Remaining data will be used for testing
    Arguments:
        tx: features
        y: labels
        ratio: specifiy data separation ratio
        seed: set at 1 if not specify
    Returns:
        x_tr: training features
        y_tr: training labels
        x_te: testing features
        y_te: testing labels
    """
    # set seed
    np.random.seed(seed)

    # split the data based on the given ratio        
    N = len(y)
    indices = np.random.permutation(N)
    size = int(np.floor(ratio*N))
    train_index = indices[:size]
    test_index = indices[size:]

    x_tr = tx[train_index]
    y_tr = y[train_index]
    x_te = tx[test_index]
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
## PRE-PROCESSING OF DATA
##******************************************************************

def process_data(tx):
    
    # In each columns, we remove nan values and replace outliers by bound
    for i in range(tx.shape[1]):
        col = np.array(tx[:, i])
        # Replace -999 value by mean
        tx[:, i] = replace_data_point_by_mean(col)
        # replace outliers by lower/higher bound
        tx[:, i] = replace_outliers_by_bound(col)
    
    ''' TO REMOVE SINCE WE DO NOT SEPARATE OUR DATA ACCORDING TO JET'''
    # remove features with no variance (mettre un treshold?)
    tx = remove_columns_with_no_variance(tx)
    
    return tx
##******************************************************************
''' TO REMOVE SINCE WE DO NOT SEPARATE OUR DATA ACCORDING TO JET '''
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
    ''' replace missing datapoint by the mean of each column
    Argument:
        col: initial feature column
    Return:
        col: processed feature column
    '''
    # If there is a nan value in the feature column
    if -999 in col:
        ind_nan = []
        compressed =[]
        for i in range(len(col)):
            
            # If the value is a nan
            if col[i] == -999:
                # store index
                ind_nan.append(i)
            else:
                # store all values which are not nan
                compressed.append(col[i])
        
        # compute the mean of all values not nan
        mean = np.mean(compressed)
        
        # replace nan value by mean
        col[col == -999] = mean
        
    return col
##******************************************************************
def replace_outliers_by_bound(col):
    ''' replace outlier values by upper/lower bound
    Argument:
        col: initial feature column
    Return:
        col: processed feature column
    '''

    # compute mean and std of the column
    std = np.std(col)
    mean = np.mean(col)
    
    # compute lower and upper bound
    low_bound = mean - 2 * std
    up_bound = mean + 2 * std
    
    # replace outliers values by respective bound
    col[col < low_bound] = low_bound
    col[col > up_bound] = up_bound
    
    return col
##******************************************************************
def standardize(x, mean = None, std = None):
    """Standardize the original data set."""
    if mean is None:
        mean = np.mean(x, axis = 0)
    x = x - mean
    if std is None:
        std = np.std(x, axis = 0)
    x = x / std
    return x, mean, std
##******************************************************************
def find_optimal_hyperparameters(
    y_train, processed_tx_train):
    '''
    Find optimal degree to expand features
    Find optimal lambda for ridge regression
    Arguments:
        y_train = labels
        processed_tx_train = features
    Returns:
        optimal_degree, optimal_lambda = optimal hyperparamaters that minimze the root mean square error loss
    '''
    
    degrees = range(4, 7)
    lambdas = np.logspace(-12, -4, 40)
    k_fold = 10
    min_loss_test = 10e6
    seed = 1
    
    k_indices = build_k_indices(y_train, k_fold, seed)

    for index_degree, degree in enumerate(degrees):

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
                min_loss_test = mean_loss_test 
                optimal_degree = degree
                optimal_lambda = lambda_
                print('degree:', optimal_degree, 
                      ' lambda:', optimal_lambda, '-> RMSE  = ', mean_loss_test)
    
    return optimal_degree, optimal_lambda


##******************************************************************
##******************************************************************
## TEST PREDICTION ON TX BY SPLITTING TX_TRAIN AND TX_TEST AND EVALUATE SCORE
##******************************************************************
def test_methods(y, tx, method):
    ''' Test all methods with optimal parameters previously found by testing. 
        Optimal parameters are initizialized within this method
        This test returns the score of each methods and is used 
        to determine which methods gives the highest score.
    Arguments:
        y: labels
        tx: features (not engineered)
        method: define which method to test
    Returns:
        scores: final score of the method
    '''
    
    #Split data for learning and testing
    ratio = 0.8
    seed = 1
    tx_train, tx_test, y_train, y_test = split_data(tx, y, ratio, seed)

    
    # Pre-processing of data 
    # -> delete nan and replace by mean column value
    # -> replace outliers by low/high bound
    processed_tx_train = process_data(tx_train)
    processed_tx_test = process_data(tx_test)

    print('Data processed')
    
    # Standardize data (needed for GD, LGD)
    standardized_tx_train, mean_tx_train, std_tx_train = standardize(processed_tx_train)
    standardized_tx_test, _, _ = standardize(processed_tx_test, mean_tx_train, std_tx_train)

    print('Data standardized')
    
    
    initial_w = np.ones(tx_train.shape[1])

    
    # Test least square GD
    if method == 'least_square_GD':
        gamma = 7*10e-4 
        max_iters = 1000
        print('least square GD learning ongoing...')
        w, loss = least_squares_GD(y_train, standardized_tx_train, initial_w, max_iters, gamma)
        score = test_score(y_test, standardized_tx_test, w)
    
    # Test least square SGD
    elif method == 'least_square_SGD':
        gamma = 0.003
        max_iters = 3000
        batch_size = 1
        print('least square SGD learning ongoing...')
        w, loss = least_squares_SGD(
                        y_train, standardized_tx_train, initial_w, batch_size, max_iters, gamma)
        score = test_score(y_test, standardized_tx_test, w)
    
    # Test least square with normal equations
    elif method == 'least_squares':
        print('least square learning ongoing...')
        w, loss = least_squares(y_train, processed_tx_train)
        score = test_score(y_test, tx_test, w)
    
    # Test ridge regression 
    elif method == 'ridge_regression':
        lambda_ = 10e-5
        print('ridge learning ongoing...')
        w, loss = ridge_regression(y_train, tx_train, lambda_)
        score = test_score(y_test, tx_test, w)
    
    else: 
        # Test logistic & regularized logistic regression
        logistic_y_train = convert_to_proba(y_train)
        logistic_y_test = convert_to_proba(y_test)
        
        if method == 'logistic_regression':
            max_iters = 4000
            gamma = 10e-7
            print('logistic regression learning ongoing...')
            w, loss = logistic_regression(
                                logistic_y_train, standardized_tx_train, initial_w,max_iters, gamma)
            score = test_logistic_score(logistic_y_test, standardized_tx_test, w)
       
        elif method == 'reg_logistic_regression':
            max_iters = 2000
            gamma = 10e-7 
            lambda_ = 0.1
            print('reg logistic regression learning ongoing...')
            w, loss = reg_logistic_regression(
                                            logistic_y_train, 
                                            standardized_tx_train, 
                                            lambda_, 
                                            initial_w, 
                                            max_iters, 
                                            gamma) 
            
            score = test_logistic_score(logistic_y_test, standardized_tx_test, w)
        
        else:    
            
            'Method not computed'
        '''
        elif method == 'poly logistic regression':
            max_iters = 4000
            gamma = 10e-3
            
            
            # Standardize data (needed for GD, LGD)
            standardized_tx_train, mean_tx_train, std_tx_train = standardize(processed_tx_train)
        
            standardized_tx_test, _, _ = standardize(processed_tx_test, mean_tx_train, std_tx_train)
            
            # Test optimized ridge regression with 
            # optimal degree = 6
            poly_tx_train = build_poly(standardized_tx_train, 6)
            poly_tx_test = build_poly(standardized_tx_test, 6)
            
            
            print('Poly built')
            
            initial_w = np.ones(poly_tx_train.shape[1])
            
            
            
            print('poly logistic regression learning ongoing...')
            w, loss = logistic_regression(
                                logistic_y_train, poly_tx_train, initial_w,max_iters, gamma)
            
            score = test_logistic_score(logistic_y_test, poly_tx_test, w)
        '''

    return score
##******************************************************************
def test_score(y, tx, w):
    """
    
    """
    y_pred = tx.dot(w)
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1
    score = np.sum(y_pred == y) / float(len(y_pred))
    print('Score: %', score * 100)
    return score
##******************************************************************
def test_logistic_score(y, tx, w):
    """
    
    """
    y_pred = tx.dot(w)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    score = np.sum(y_pred == y) / float(len(y_pred))
    print('Score: %', score * 100)
    return score

##******************************************************************
##******************************************************************
## JET NUM SEPARATION
##******************************************************************
def create_batch(jet_num, tx, y= None):
    
    # Jet num == column 22

    # keep track of index
    
    index = np.where(x[:,22]== jet_num)
    
    tx =tx[index]
    y=y[index]
    
    return tx, y



