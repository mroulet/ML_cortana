import numpy as np

from proj1_helpers import *
from implementations import *
from helpers import *
from data_manager import *


def main (methods):
    """
      Main method, creating a submission .csv of predictions with the best model chosen
      here we use the ridge_regression, with polynomiale construction of parameters and cross-terms
      parameters used for ridge-regression, i.e. lambdas and degres are taken from another method
      that finds optimals parameters
    Arguments:
        methods: method chosen for the best model
    """


    #load datas
    """
    y: labels for train
    tx: features matrix for train
    y_test: labels for test
    tx_test: features matrix for test
    """
    DATA_TRAIN_PATH = '../data/train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

    DATA_TEST_PATH = '../data/test.csv'
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    print("data loaded")

    #best parameters found from learning
    gammas = [1e-12,7.443803013251697e-10,1.1253355826007645e-10,1.7012542798525856e-11]
    degres = [6,7,6,6]


    pred = []
    id_t =[]

    #creation of jet_num
    jets_y, jets_tX, _ = cat_variables(y, tX, ids)
    jets_y_test, jets_tX_test, id_test = cat_variables(y_test, tX_test, ids_test)

    #options of pre-processing
        #nandel -> delete nan values (-999.)
        #nanmed -> replace -999. with mean
        #bound -> eliminate outliers
        #std -> standardize data set
        #prb -> change y into probability values
        #zerovar -> eliminate columns with no variances
    My_options = ['nanmed', 'bound', 'zerovar','std']

    #iteration in each jet_num batch
    for ind in range(len(jets_y)):
        print('Analyzing jet {}'.format(ind))

        #pre-process data (train and test)
        jets_y[ind], jets_tX[ind] = process_data(jets_y[ind], jets_tX[ind], My_options)
        jets_y_test[ind], jets_tX_test[ind] = process_data(jets_y_test[ind], jets_tX_test[ind], My_options)

        #construction of the final features containing polynomiale features and cross-terms
        final_tX_train = np.c_[build_poly(jets_tX[ind],degres[ind]), build_cross_terms(jets_tX[ind])]
        final_tX_test = np.c_[build_poly(jets_tX_test[ind],degres[ind]), build_cross_terms(jets_tX_test[ind])]

        #parameters token by the chosen method
        param = [jets_y[ind], final_tX_train, gammas[ind]]

        #create weights and losses
        w,loss = test_methods(methods, param)

        #creation of predictions
        pred_test = predict(final_tX_test,w)

        #lists of prediction and ids for each jet batch
        pred.append(pred_test)
        id_t.append(id_test[ind])



    # concatenation lists of ids and predictions for each jet batch in a single well arranged array
    pred =np.concatenate(pred, 0)
    id_t =np.concatenate(id_t,0)


    #submission
    OUTPUT_PATH = "../data/submissioncortana.csv"
    create_csv_submission(id_t, pred, OUTPUT_PATH)
    print('Submission saved as ', OUTPUT_PATH)







main(ridge_regression)
