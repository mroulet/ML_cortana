# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def plot_my_values(weights, y, tX, degrees, gammas, lambdas, logistic):
    
    if logistic:
        plot_my_values_log(weights, y, tX, degrees, gammas, lambdas)
        return
    
    deg = False
    gam = False
    lamb = False
    
    if degrees :
        deg = True
    if gammas :
        gam = True
    if lambdas :
        lamb = True
    if (not deg) and (not gam) and (not lamb) :
        raise Exception('No parameters were entered in plot_my_values')
    
    accuracy = []
    
    if deg and (not gam) and (not lamb):
        for ind, w in enumerate(weights):
            tX_poly = build_poly(tX, degrees[0][ind])
            accuracy.append(test_score(y, tX_poly, w))
        plotML_degree(accuracy, degrees[0])
    
    elif (not deg) and gam and (not lamb):
        for w in weights:
            accuracy.append(test_score(y, tX, w))
        plotML_gamma(accuracy, gammas[0])
    
    elif (not deg) and (not gam) and lamb:
        for w in weights:
            accuracy.append(test_score(y, tX, w))
        plotML_lambda(accuracy, lambdas[0])
    
    else :
        raise Exception('Sorry, this program is not ready for 3D and 4D yet')
        

def plot_my_values_log(weights, y, tX, degrees, gammas, lambdas):
    
    deg = False
    gam = False
    lamb = False
    
    if degrees :
        if len(degrees[0]) > 1:
            deg = True
    if gamma :
        gam = True
    if lambdas :
        lamb = True
        
    if (not deg) and (not gam) and ( not lamb) :
        raise Exception('No parameters were entered in plot_my_values')
    
    accuracy = []
    
    if deg and (not gam) and (not lamb):
        for w in weights:
            tX_poly = build_poly(tX, degrees[0][ind])
            accuracy.append(test_score(y, tX_poly, w))
        plotML_degree(accuracy, degrees[0])
    
    elif (not deg) and gam and (not lamb):
        for w in weights:
            accuracy.append(test_logistic_score(y, tX, w))
        plotML_gamma(accuracy, gammas[0])
    
    elif (not deg) and (not gam) and lamb:
        for w in weights:
            accuracy.append(test_logistic_score(y, tX, w))
        plotML_lambda(accuracy, lambdas[0])

    else :
        raise Exception('Sorry, this program is not ready for 3D and 4D yet')
        

def plotML_gamma(accuracy, gammas):
    """Visualization of the curves of accuracy vs gamma"""
    plt.semilogx(gammas, accuracy, marker=".", label='curve')
    plt.xlabel("gamma")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs gamma")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("Gammacuracy")

def plotML_degree(accuracy, degrees):
    """Visualization of the curves of accuracy vs degree"""
    plt.plot(degrees, accuracy, marker=".", label='curve')
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs degree")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("degreacuracy")

def plotML_lambda(accuracy, lambdas):
    """Visualization of the curves of accuracy vs degree"""
    plt.semilogx(lambdas, accuracy, marker=".", label='curve')
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs lambda")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("lambdacuracy")
    
    
def plotML_3D(accuracy, degrees, gammas):
    
    #import matplotlib.pyplot as plt
    #from matplotlib import cm
    #from matplotlib.ticker import LinearLocator, FormatStrFormatter
    #import numpy as np


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    degrees, gammas = np.meshgrid(degrees, gammas)


    # Plot the surface.
    surf = ax.plot_surface(degrees, gammmas, accuracy, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
