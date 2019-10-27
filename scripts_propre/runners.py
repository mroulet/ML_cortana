# -*- coding: utf-8 -*-
"""Big codes to run."""
import numpy as np
from helpers import *
from data_manager import *
from implementations import *

def optimization(y, tX, method, degrees, gammas, lambdas, grad):
    
    """ Iterate you method on all your different parameters and return the weights and losses 
    Arguments:
        y: labels
        tx: features
        method: method to be used
        degrees : list of degrees (can be empty)
        gammas : list of gammas (can be empty)
        lambdas : list of lambdas (can be empty)
        grad : contains the parameters for a gradient learning 
                [bool(True if you want to do a gradient learning), w_init, max_iters] 
    Returns:
        weights: list of all the computed weights
        losses: list of all losses 
    """
        
    # If your method uses gradients the function varies slighty, check optimize_grad
    if grad[0]:
        weights, losses = optimization_grad(y, tX, method, degrees, gammas, lambdas, grad)
        return weights, losses
    

    # list that will regroup all your non empty parameters
    param = []
    deg = False
    
    # Combine the parameters together, the order of the parameters correspond
    # to the same order as the arguments are given in the methods
    # When adding a new method be aware of the order of your arguments
    if degrees :
        deg = True
        param.append(degrees)
    if gammas :
        param.append(gammas)
    if lambdas :
        param.append(lambdas)

    # Initialize parameters, losses and weights
    parameters = []    
    losses = []
    weights = []
    
   
    # Do an optimization depending on the number of parameters
    # For each iteration: - create build poly (if degree option on)
    #                     - stack values in parameters
    #                     - run the method with parameters
    #                     - save weight and loss obtained
    
    
    # If 0 list of parameters (= param empty)
    if not param:
        parameters = [y, tX]
        w, loss = test_methods(method, parameters)
        losses.append(loss)
        weights.append(w)
            
    # If 1 list of parameter was given
    if len(param) == 1:
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                print('runing degree {}'.format(param[0][0][ind1]))
                parameters = [y, tX_poly]
                w, loss = test_methods(method, parameters)
                losses.append(loss)
                weights.append(w)
            else:
                parameters = [y, tX, param[0][0][ind1]]
                w, loss = test_methods(method, parameters)
                losses.append(loss)
                weights.append(w)
                
    # If 2 lists of parameter were given
    if len(param) == 2:
        print('runing with two types of parameter...')
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                print('runing degree {}'.format(param[0][0][ind1]))
                for ind2 in range(len(param[1][0])):
                    parameters = [y, tX_poly, param[1][0][ind2]]
                    w, loss = test_methods(method, parameters)
                    losses.append(loss)
                    weights.append(w)
            else:
                for ind2 in range(len(param[1][0])):
                    parameters = [y, tX, param[0][0][ind1], param[1][0][ind2]]
                    w, loss = test_methods(method, parameters)
                    losses.append(loss)
                    weights.append(w)
                    
    # If 3 lists of parameter were given
    if len(param) == 3:
        print('runing with three types of parameter...')
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                print('runing degree {}'.format(param[0][ind1]))
                for ind2 in range(len(param[1][0])):
                    for ind3 in range(len(param[2][0])):
                        parameters = [y, tX_poly, param[1][0][ind2], param[2][0][ind3]]
                        w, loss = test_methods(method, parameters)
                        losses.append(loss)
                        weights.append(w)
            else:
                for ind2 in range(len(param[1][0])):
                    for ind3 in range(len(param[2][0])):
                        parameters = [y, tX, param[0][0][ind1], param[1][0][ind2], param[2][0][ind3]]
                        w, loss = test_methods(method, parameters)
                        losses.append(loss)
                        weights.append(w)
    
    return weights, losses



def optimization_grad(y, tX, method, degrees, gammas, lambdas, grad):
    """ Iterate you method on all your different parameters and return the weights and losses 
    Arguments:
        y: labels
        tx: features
        method: method to be used
        degrees : list of degrees (can be empty)
        gammas : list of gammas (can be empty)
        lambdas : list of lambdas (can be empty)
        grad : contains the parameters for a gradient learning 
                [bool(True if you want to do a gradient learning), w_init, max_iters] 
    Returns:
        weights: list of all the computed weights
        losses: list of all losses
    """
    
    param = []
    deg = False
    
    # list that will regroup all your non empty parameters
    if degrees :
        deg = True
        param.append(degrees)
    if gammas :
        param.append(gammas)
    if lambdas :
        param.append(lambdas)
     
    # Initialize parameters that will be used in method, losses and weights
    parameters = []    
    losses = []
    weights = []
    

    
    # Create w_initial if it is not already
    if not deg and not grad[1]:
        grad[1] = [np.zeros(tX.shape[1])]
       
    
    # Do an optimization depending on the number of parameters
    # For each iteration: - create build poly (if degree option on)
    #                     - stack values in parameters
    #                     - run the method with parameters
    #                     - save weight and loss obtained
    
    
    # If 1 lists of parameter were given
    if len(param) == 1:
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                grad[1] = [np.zeros(tX_poly.shape[1][0])]
                print('runing degree {}'.format(param[0][0][ind1]))
                parameters = [y, tX_poly, grad[1][0], grad[2]]
                w, loss = test_methods(method, parameters)
                losses.append(loss)
                weights.append(w)
            else:
                parameters = [y, tX, grad[1][0], grad[2], param[0][0][ind1]]
                w, loss = test_methods(method, parameters)
                losses.append(loss)
                weights.append(w)
                
    # If 2 lists of parameter were given
    if len(param) == 2:
        print('runing with two types of parameter...')
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                grad[1] = [np.zeros(tX_poly.shape[1])]
                print('runing degree {}'.format(param[0][0][ind1]))
                for ind2 in range(len(param[1][0])):
                    parameters = [y, tX_poly, grad[1][0], grad[2], param[1][0][ind2]]
                    w, loss = test_methods(method, parameters)
                    losses.append(loss)
                    weights.append(w)
            else:
                for ind2 in range(len(param[1][0])):
                    parameters = [y, tX, grad[1][0], grad[2], param[0][0][ind1], param[1][0][ind2]]
                    w, loss = test_methods(method, parameters)
                    losses.append(loss)
                    weights.append(w)
                    
    # If 3 lists of parameter were given
    if len(param) == 3:
        print('runing with three types of parameter...')
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                grad[1] = [np.zeros(tX_poly.shape[1])]
                print('runing degree {}'.format(param[0][0][ind1]))
                for ind2 in range(len(param[1][0])):
                    for ind3 in range(len(param[2][0])):
                        parameters = [y, tX_poly, grad[1][0], grad[2], param[1][0][ind2], param[2][0][ind3]]
                        w, loss = test_methods(method, parameters)
                        losses.append(loss)
                        weights.append(w)
            else:
                for ind2 in range(len(param[1][0])):
                    for ind3 in range(len(param[2][0])):
                        parameters = [y, tX, grad[1][0], grad[2], param[0][0][ind1], param[1][0][ind2], param[2][0][ind3]]
                        w, loss = test_methods(method, parameters)
                        losses.append(loss)
                        weights.append(w)
    
    return weights, losses


def optimization_cross(y, tX, method, degrees, gammas, lambdas, grad):
    
    """ Iterate you method on all your different parameters and return the weights and losses 
    Arguments:
        y: labels
        tx: features
        method: method to be used
        degrees : list of degrees (can be empty)
        gammas : list of gammas (can be empty)
        lambdas : list of lambdas (can be empty)
        grad : contains the parameters for a gradient learning 
                [bool(True if you want to do a gradient learning), w_init, max_iters] 
    Returns:
        weights: list of all the computed weights
        losses: list of all losses 
    """
        
    # If your method uses gradients the function varies slighty, check optimize_grad
    if grad[0]:
        weights, losses = optimization_cross_grad(y, tX, method, degrees, gammas, lambdas, grad)
        return weights, losses
    

    # list that will regroup all your non empty parameters
    param = []
    deg = False
    
    # Combine the parameters together, the order of the parameters correspond
    # to the same order as the arguments are given in the methods
    # When adding a new method be aware of the order of your arguments
    if degrees :
        deg = True
        param.append(degrees)
    if gammas :
        param.append(gammas)
    if lambdas :
        param.append(lambdas)

    # Initialize parameters, losses and weights
    parameters = []    
    losses = []
    weights = []
    
   
    # Do an optimization depending on the number of parameters
    # For each iteration: - create build poly (if degree option on)
    #                     - stack values in parameters
    #                     - run the method with parameters
    #                     - save weight and loss obtained
    
    
    # If 0 list of parameters (= param empty)
    if not param:
        parameters = [y, tX]
        w, loss = cross_validation(method, parameters)
        losses.append(loss)
        weights.append(w)
            
    # If 1 list of parameter was given
    if len(param) == 1:
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                print('runing degree {}'.format(param[0][0][ind1]))
                parameters = [y, tX_poly]
                w, loss = cross_validation(method, parameters)
                losses.append(loss)
                weights.append(w)
            else:
                parameters = [y, tX, param[0][0][ind1]]
                w, loss = cross_validation(method, parameters)
                losses.append(loss)
                weights.append(w)
                
    # If 2 lists of parameter were given
    if len(param) == 2:
        print('runing with two types of parameter...')
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                print('runing degree {}'.format(param[0][0][ind1]))
                for ind2 in range(len(param[1][0])):
                    parameters = [y, tX_poly, param[1][0][ind2]]
                    w, loss = cross_validation(method, parameters)
                    losses.append(loss)
                    weights.append(w)
            else:
                for ind2 in range(len(param[1][0])):
                    parameters = [y, tX, param[0][0][ind1], param[1][0][ind2]]
                    w, loss = cross_validation(method, parameters)
                    losses.append(loss)
                    weights.append(w)
                    
    # If 3 lists of parameter were given
    if len(param) == 3:
        print('runing with three types of parameter...')
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                print('runing degree {}'.format(param[0][ind1]))
                for ind2 in range(len(param[1][0])):
                    for ind3 in range(len(param[2][0])):
                        parameters = [y, tX_poly, param[1][0][ind2], param[2][0][ind3]]
                        w, loss = cross_validation(method, parameters)
                        losses.append(loss)
                        weights.append(w)
            else:
                for ind2 in range(len(param[1][0])):
                    for ind3 in range(len(param[2][0])):
                        parameters = [y, tX, param[0][0][ind1], param[1][0][ind2], param[2][0][ind3]]
                        w, loss = cross_validation(method, parameters)
                        losses.append(loss)
                        weights.append(w)
    
    return weights, losses



def optimization_cross_grad(y, tX, method, degrees, gammas, lambdas, grad):
    """ Iterate you method on all your different parameters and return the weights and losses 
    Arguments:
        y: labels
        tx: features
        method: method to be used
        degrees : list of degrees (can be empty)
        gammas : list of gammas (can be empty)
        lambdas : list of lambdas (can be empty)
        grad : contains the parameters for a gradient learning 
                [bool(True if you want to do a gradient learning), w_init, max_iters] 
    Returns:
        weights: list of all the computed weights
        losses: list of all losses
    """
    
    param = []
    deg = False
    
    # list that will regroup all your non empty parameters
    if degrees :
        deg = True
        param.append(degrees)
    if gammas :
        param.append(gammas)
    if lambdas :
        param.append(lambdas)
     
    # Initialize parameters that will be used in method, losses and weights
    parameters = []    
    losses = []
    weights = []
    

    
    # Create w_initial if it is not already
    if not deg and not grad[1]:
        grad[1] = [np.zeros(tX.shape[1])]
       
    
    # Do an optimization depending on the number of parameters
    # For each iteration: - create build poly (if degree option on)
    #                     - stack values in parameters
    #                     - run the method with parameters
    #                     - save weight and loss obtained
    
    
    # If 0 list of parameters (= param empty)
    if not param:
        parameters = [y, tX, grad[1][0], grad[2]]
        w, loss = cross_validation(method, parameters)
        losses.append(loss)
        weights.append(w)
        
    # If 1 lists of parameter were given
    if len(param) == 1:
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                grad[1] = [np.zeros(tX_poly.shape[1][0])]
                print('runing degree {}'.format(param[0][0][ind1]))
                parameters = [y, tX_poly, grad[1][0], grad[2]]
                w, loss = cross_validation(method, parameters)
                losses.append(loss)
                weights.append(w)
            else:
                parameters = [y, tX, grad[1][0], grad[2], param[0][0][ind1]]
                w, loss = cross_validation(method, parameters)
                losses.append(loss)
                weights.append(w)
                
    # If 2 lists of parameter were given
    if len(param) == 2:
        print('runing with two types of parameter...')
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                grad[1] = [np.zeros(tX_poly.shape[1])]
                print('runing degree {}'.format(param[0][0][ind1]))
                for ind2 in range(len(param[1][0])):
                    parameters = [y, tX_poly, grad[1][0], grad[2], param[1][0][ind2]]
                    w, loss = cross_validation(method, parameters)
                    losses.append(loss)
                    weights.append(w)
            else:
                for ind2 in range(len(param[1][0])):
                    parameters = [y, tX, grad[1][0], grad[2], param[0][0][ind1], param[1][0][ind2]]
                    w, loss = cross_validation(method, parameters)
                    losses.append(loss)
                    weights.append(w)
                    
    # If 3 lists of parameter were given
    if len(param) == 3:
        print('runing with three types of parameter...')
        for ind1 in range(len(param[0][0])):
            if deg == True:
                tX_poly = build_poly(tX, param[0][0][ind1])
                grad[1] = [np.zeros(tX_poly.shape[1])]
                print('runing degree {}'.format(param[0][0][ind1]))
                for ind2 in range(len(param[1][0])):
                    for ind3 in range(len(param[2][0])):
                        parameters = [y, tX_poly, grad[1][0], grad[2], param[1][0][ind2], param[2][0][ind3]]
                        w, loss = cross_validation(method, parameters)
                        losses.append(loss)
                        weights.append(w)
            else:
                for ind2 in range(len(param[1][0])):
                    for ind3 in range(len(param[2][0])):
                        parameters = [y, tX, grad[1][0], grad[2], param[0][0][ind1], param[1][0][ind2], param[2][0][ind3]]
                        w, loss = cross_validation(method, parameters)
                        losses.append(loss)
                        weights.append(w)
    
    return weights, losses


##******************************************************************
## TEST PREDICTION ON TX BY SPLITTING TX_TRAIN AND TX_TEST AND EVALUATE SCORE
##******************************************************************
def test(y, tx, method):
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
    y_train, tx_train, y_test, tx_test = split_data(y, tx, ratio, seed)

    

    """Reminder for My_options (processing data):
     nandel -> delete nan values (-999.)
     nanmed -> replace -999. with mean
     bound -> eliminate outliers
     std -> standardize data set
     prb -> change y into probability values
     zerovar -> eliminate columns with no varinaces
    """
    
    initial_w = np.ones(tx_train.shape[1])

    # steps for any function : -> process data
    #                          -> initialize parameters
    #                          -> run method
    #                          -> evaluate score
    
    # Test least square GD
    if method == 'least_square_GD':
        
        My_options = ['nanmed', 'bound', 'std']
        y_tr, tx_tr = process_data(y_train, tx_train, My_options)
        y_te, tx_te = process_data(y_test, tx_test, My_options)
        
        gamma = 7*10e-4 
        max_iters = 1000
        
        print('least square GD learning ongoing...')
        w, loss = least_squares_GD(y_tr, tx_tr, initial_w, max_iters, gamma)
        score = test_score(y_te, tx_te, w)
    
    # Test least square SGD
    elif method == 'least_square_SGD':
        
        My_options = ['nanmed', 'bound', 'std']
        y_tr, tx_tr = process_data(y_train, tx_train, My_options)
        y_te, tx_te = process_data(y_test, tx_test, My_options)
        
        gamma = 0.003
        max_iters = 3000
        batch_size = 1
        
        print('least square SGD learning ongoing...')
        w, loss = least_squares_SGD(y_tr, tx_tr, initial_w, batch_size, max_iters, gamma)
        score = test_score(y_te, tx_te, w)
    
    # Test least square with normal equations
    elif method == 'least_squares':
        
        My_options = ['nanmed', 'bound']
        y_tr, tx_tr = process_data(y_train, tx_train, My_options)
        y_te, tx_te = process_data(y_test, tx_test, My_options)
        
        print('least square learning ongoing...')
        w, loss = least_squares(y_tr, tx_tr)
        score = test_score(y_te, tx_te, w)
    
    # Test ridge regression 
    elif method == 'ridge_regression':
        
        My_options = ['nanmed', 'bound']
        y_tr, tx_tr = process_data(y_train, tx_train, My_options)
        y_te, tx_te = process_data(y_test, tx_test, My_options)
        
        lambda_ = 10e-5
        
        print('ridge learning ongoing...')
        w, loss = ridge_regression(y_tr, tx_tr, lambda_)
        score = test_score(y_te, tx_te, w)
    
    # Test logistic regression
    elif method == 'logistic_regression':

        My_options = ['nanmed', 'bound', 'std', 'prb']
        y_tr, tx_tr = process_data(y_train, tx_train, My_options)
        y_te, tx_te = process_data(y_test, tx_test, My_options)
        
        max_iters = 4000
        gamma = 10e-7
            
        print('logistic regression learning ongoing...')
        w, loss = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
        score = test_logistic_score(y_tr, tx_tr, w)
       
    # Test regularized logistic regression
    elif method == 'reg_logistic_regression':
            
        My_options = ['nanmed', 'bound', 'std', 'prb']
        y_tr, tx_tr = process_data(y_train, tx_train, My_options)
        y_te, tx_te = process_data(y_test, tx_test, My_options)
        
        max_iters = 2000
        gamma = 10e-7 
        lambda_ = 0.1
            
        print('reg logistic regression learning ongoing...')
        w, loss = reg_logistic_regression( y_tr, tx_tr, lambda_, initial_w, max_iters, gamma) 
        score = test_logistic_score(y_te, tx_te, w)
        
    else :
        raise Exception('Invalid method name')
        
    return score