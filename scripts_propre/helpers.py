# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np

## ******************************************************************
def sigmoid(t):
    """apply sigmoid function on t."""
    return 1. / (1+np.exp(-t))

## ******************************************************************
def compute_logistic_loss(y, tx, w,):
    """compute the cost by negative log likelihood."""
    a = np.sum(np.log(1+np.exp(tx.dot(w))))
    b = y.T.dot(tx.dot(w))
    
    return a - b
    
## ******************************************************************
def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(sigmoid(tx.dot(w))-y)


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
    
    return -(tx.T.dot(e)) / e.size


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
        #if tx.shape[1] > 1 and col != tx.shape[1] - 1:
        #   poly_tx = np.c_[poly_tx, np.ones((len(tx), 1))] 
     
    return poly_tx


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
def select_values(y, tX, k_indices, k) :
    
    y_te, tX_te = y[k_indices[k]], tX[k_indices[k]]
    k_indices = np.ma.array(k_indices, mask=False)      #Create mask on k_indices
    k_indices.mask[k] = True                            #Hide the kth line of k_indices with mask
    k_indices_tr = np.ravel(k_indices).compressed()     #ravel->transform indices into array/compressed->eliminate masked values
    y_tr, tX_tr = y[k_indices_tr], tX[k_indices_tr]
    
    return y_tr, tX_tr, y_te, tX_te

##******************************************************************
def weight_mean(weights):
    """ Mean weight of a collection of weights. """
    w = np.zeros(weights[0].shape)
    W = np.stack(weights, axis=0)
    for ind in range(w.shape[0]):
        w[ind] = np.mean(W[:, ind])
    return w

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
def predict(tx, w):
    """
    
    """
    y_pred = tx.dot(w)
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1
    
    return y_pred
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
def test_methods(method, parameters):
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
    
    if (len(parameters) == 2):
        w, loss = method(parameters[0], parameters[1])
    elif (len(parameters) == 3):
        w, loss = method(parameters[0], parameters[1], parameters[2])
    elif (len(parameters) == 4):
        w, loss = method(parameters[0], parameters[1], parameters[2], parameters[3])
    elif (len(parameters) == 5):
        w, loss = method(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4])
    else :
        raise Exception('Uncorrect number of parameters in cross_validation')

    return w, loss


def cross_validation(method, parameters):
    
    k_fold = 4
    seed = 1
    k_indices = build_k_indices(parameters[0], k_fold, seed)
    losses = []
    weights = []
    
    for k in range(k_fold):
        y_tr, tX_tr, y_te, tX_te = select_values(parameters[0], parameters[1], k_indices, k)
        
        if (len(parameters) == 2):
            w, loss = method(parameters[0], parameters[1])
        elif (len(parameters) == 3):
            w, loss = method(parameters[0], parameters[1], parameters[2])
        elif (len(parameters) == 4):
            w, loss = method(parameters[0], parameters[1], parameters[2], parameters[3])
        else :
            raise Exception('Uncorrect number of parameters in cross_validation')
        
        losses.append(loss)
        weights.append(w)

    return weight_mean(weights), np.mean(loss)


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