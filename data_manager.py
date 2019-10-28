# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np


## ******************************************************************
def process_data(y, tX, My_options):
    """
    pre-processing of the datas
    Arguments :
        y: labels
        tX: features matrix
        My_options: chosen processing options
    Return :
        y and tX modified

    """
    # List of the possible options
    options = ['nandel',
               'nanmed',
               'bound',
               'std',
               'prb',
               'zerovar']

    # Check if all the given options exits, else return Exception
    for option in My_options :
        if (option in options) == False :
            raise Exception("{} not available, check list of options".format(option))


    # Apply options that were given
    y_out = y
    tX_out = tX

    if "zerovar" in My_options :
        tX_out = remove_columns_with_no_variance(tX_out)
    if "nandel" in My_options :
        y_out, tX_out = delete_nan(y_out, tX_out)
    if "nanmed" in My_options :
        tX_out = nan_to_median(tX_out)
    if "bound" in My_options :
        tX_out = replace_outliers_by_bound(tX_out)
    if "std" in My_options :
        tX_out = standardize(tX_out)
    if "prb" in My_options :
        y_out = convert_to_proba(y_out)


    return y_out, tX_out


## ******************************************************************
def split_data(y, x, ratio, seed=1):
    """
        split the dataset based on the split ratio.
        Arguments :
            y: labels
            x: features matrix
            ratio: ratio chosen between the train and the test set
        Returns:
            _tr : data set for train
            _te : data set for test


    """
    # set seed
    np.random.seed(seed)

    # shuffle indices and seperate depending on the ratio
    indices = np.random.permutation(x.shape[0])
    seperation = int(x.shape[0]*ratio)
    training_idx, test_idx = indices[:seperation], indices[seperation:]

    y_tr, y_te = y[training_idx], y[test_idx]
    x_tr, x_te = x[training_idx, :], x[test_idx, :]

    return y_tr, x_tr, y_te, x_te


## ******************************************************************
def nan_to_median(tX):
    """ Replace -999. values with median

    Argument:
        tX: data set
    Return:
        tX : modified tX
    """
    # Choose best option, eliminate value ? median ? mean ? ...
    for ind in range(tX.shape[1]):
        if -999. in tX[:, ind]:
            tX[:, ind] = np.where(tX[:, ind] == -999., np.nan, tX[:, ind]) # transform -999. into nan for easy median calculation
            median = np.nanmedian(tX[:, ind])
            tX[:, ind] = np.where(np.isnan(tX[:, ind]), median, tX[:, ind]) # replace with median
    return tX


## ******************************************************************
def delete_nan(y, tX):
    """ Delete -999. values
        Argument:
            tX: data set
            y: labels
        Return:
            tX : modified tX
            y: modified y

     """
    tX = np.where(tX == -999., np.nan, tX)

    #np.isnan gives a boolean matrix of size tX with true when the element is a nan
    #np.any returns a array where each element shows if there was at least one true along the axis 1
    index = ~np.isnan(tX).any(axis=1)

    return y[index], tX[index]

## ******************************************************************
def convert_to_proba(y):
    """ Transform {-1, 1} data into {0, 1} data.
        Argument:
            y: labels
        Return:
            y: modified y
     """
    # Not sure if this is really needed
    y[y == -1] = 0
    return y


## ******************************************************************
def remove_columns_with_no_variance(tx):
    """ remove columns that have no variances
        Argument:
            tX: data set
        Return:
            tX : modified tX

    """
    columns = np.full(tx.shape[1], True, dtype=bool)

    for col in range(tx.shape[1]):
        # If the columns have a unique element (all data have same value)
        if len(np.unique(tx[:,col])) == 1:
            columns[col] = False

    return tx[:, columns]


## ******************************************************************
def standardize(tX, mean_x = None, std_x = None):
    """Standardize the original data set.
        Argument:
            tX: data set
        Return:
            tX : modified tX

    """
    if mean_x is None:
        mean_x = np.mean(tX, axis = 0)
    tX = tX - mean_x
    if std_x is None:
        std_x = np.std(tX, axis = 0)
    tX = tX / std_x
    return tX #, mean_x, std_x


## ******************************************************************
def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization.
        Argument:
            tX: data set
        Return:
            tX : modified tX

    """
    x = x * std_x
    x = x + mean_x
    return x


## ******************************************************************
def cat_variables(y, tX, ids):
    """ JetNum is the only categorical variable in the data set. cat_variables() seperates
        the data in four parts depending on the value of JetNum.
        Unfortunately we do not have much knowledge on how to manage categorical variables...
            Argument:
                tX: data set
                y: labels
                ids : indexes
            Return:
                tX : modified tX
                y: modified y
                ids: indexes


        """

    col_jets = np.nan

    # Find the column that defines the jet features
    for ind in range(tX.shape[1]) :
        values = np.unique(tX[:, ind])

        if np.array_equal(values, [0., 1., 2., 3.]):
            col_jets = ind
            break

    # Raise error if no jet feature was found
    if np.isnan(col_jets):
        raise Exception('JetNumberColumnNotFound')

    jets_y = []
    jets_tX = []
    jet_ids = []

    # Create one data set for each jet Number and stacks it in a list
    for jetNum in values:
        mask = tX[:, col_jets] == jetNum
        jets_y.append(y[mask])
        jets_tX.append(tX[mask])
        jet_ids.append(ids[mask])

    return jets_y, jets_tX, jet_ids


## ******************************************************************
def replace_outliers_by_bound(tX):
    ''' replace outlier values by upper/lower bound
    Argument:
        tX: data set
    Return:
        tX : modified tX

    This function should only be used when nan/-999 are not here anymore
    '''

    for col in range(tX.shape[1]) :

        # compute mean and std of the column
        mean = np.mean(tX[:, col])
        std = np.std(tX[:, col])

        # compute lower and upper bound
        low_bound = mean - 2 * std
        up_bound = mean + 2 * std

        # replace outliers values by respective bound
        tX[:, col] = np.where(tX[:, col] < low_bound, low_bound, tX[:, col])
        tX[:, col] = np.where(tX[:, col] > up_bound, up_bound, tX[:, col])

    return tX
