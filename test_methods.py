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

    

    '''Reminder for My_options (processing data):
     nandel -> delete nan values (-999.)
     nanmed -> replace -999. with mean
     bound -> eliminate outliers
     std -> standardize data set
     prb -> change y into probability values
     zerovar -> eliminate columns with no varinaces
     '''
    
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
    
    elif method == 'logistic_regression': 
        # Test logistic & regularized logistic regression
        My_options = ['nanmed', 'bound', 'std', 'prb']
        y_tr, tx_tr = process_data(y_train, tx_train, My_options)
        y_te, tx_te = process_data(y_test, tx_test, My_options)
        
        max_iters = 4000
        gamma = 10e-7
            
        print('logistic regression learning ongoing...')
        w, loss = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
        score = test_logistic_score(y_tr, tx_tr, w)
       
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
        
    else:
        raise Exception('Invalid function name in test_methods')
          
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