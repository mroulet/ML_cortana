# -*- coding: utf-8 -*-
"""Big codes to run."""
import numpy as np
from helpers import *

def optimization(y, tX, method, degrees, gammas, lambdas, grad):
    """ Iterate you method on all your different parameters and return the weights and losses """
    
    # If your method uses gradients the function varies slighty, check optimize_grad
    if grad[0] :
        weights, losses = optimization_grad(y, tX, method, degrees, gammas, lambdas, grad)
        return weights, losses
       
    
    param = []
    deg = False
    
    # Combine the parameters together (Note that if degrees is not empty it is always placed first)
    if degrees :
        deg = True
        param.append(degrees)
    if gammas :
        param.append(gammas)
    if lambdas :
        param.append(lambdas)
    if not param :
        raise Exception('No parameters were entered in optimization')
     
    # Initialize parameters that will be used in method, losses and weights
    parameters = []    
    losses = []
    weights = []
    
   
    # Do an optimization depending on the number of parameters
    # If one type of parameter was given
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
                
    # If two types of parameter were given
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
                    
    # If three types of parameter were given
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
    """ Iterate you method on all your different parameters and return the weights and losses """
    
    param = []
    deg = False
    
    # Combine the parameters together (Note that if degrees is not empty it is always placed first)
    if degrees :
        deg = True
        param.append(degrees)
    if gammas :
        param.append(gammas)
    if lambdas :
        param.append(lambdas)
    if not param :
        raise Exception('No parameters were entered in optimization')
     
    # Initialize parameters that will be used in method, losses and weights
    parameters = []    
    losses = []
    weights = []
    
    # Create w_initial if it is not already
    if not deg and not grad[1]:
        grad[1] = [np.zeros(tX.shape[1])]
        
   
    # Do an optimization depending on the number of parameters
    # If one type of parameter was given
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
                
    # If two types of parameter were given
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
                    
    # If three types of parameter were given
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