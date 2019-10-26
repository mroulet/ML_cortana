# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


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
    plt.plot(gammas, accuracy, marker=".")
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs degree")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("degreacuracy")

    

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
