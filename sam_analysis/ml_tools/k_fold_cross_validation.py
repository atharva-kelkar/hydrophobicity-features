"""
k_fold_cross_validation.py
This script contains code to run k-fold cross validation

CREATED ON: 01/05/2021

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT OS AND SYS
import os
## IMPORT NUMPY
import numpy as np  # Used to do math functions
## IMPORT REGRESSION FUNCTIONS FROM SKLEARN
from sklearn.linear_model import LinearRegression
from sklearn import metrics

## IMPORT GLOBAL INPUTS
from sam_analysis.ml_tools.globals import GROUP_LIST, X_DATA_LIST, DATA_LABELS,        \
                                          TRAINING_GROUPS, TESTING_GROUPS,             \
                                          NH2_TRAINING_GROUPS, NH2_TESTING_GROUPS,     \
                                          CONH2_TRAINING_GROUPS, CONH2_TESTING_GROUPS, \
                                          OH_TRAINING_GROUPS, OH_TESTING_GROUPS
## IMPORT DATA TOOLS
from sam_analysis.ml_tools.data_tools import rescale_data, compile_ml_data
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl
## IMPORT CHECK SERVER PATH
from sam_analysis.core.check_tools import check_server_path
## IMPORT PLOTTING TOOLS
from sam_analysis.ml_tools.plotting import plot_parity, plot_saliency, plot_multi_saliency
                                           
#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## N-SAMPLES
    n_samples = 3
    
    ## WORKING DIRECTORY
    working_dir = r"/mnt/r/python_projects/sam_analysis/sam_analysis/raw_data"
    working_dir = check_server_path( working_dir )
    
    ## FEATURES FILE
    feature_pkl = r"unbiased_regression_data.pkl"
#    feature_pkl = r"unbiased_regression_data_wc.pkl"
    
    ## LABELS FILE
    label_pkl = r"indus_regression_data.pkl"
    
    ## LOAD DATA
    path_X_pkl = os.path.join( working_dir, feature_pkl )
    X_raw_data = load_pkl( path_X_pkl )
    
    ## LOAD LABEL DATA
    path_y_pkl = os.path.join( working_dir, label_pkl )
    y_raw_data = load_pkl( path_y_pkl )
                      
    ## COMPILE DATA
    XX, yy = compile_ml_data( group_list    = GROUP_LIST,
                              X_raw         = X_raw_data, 
                              y_raw         = y_raw_data, 
                              X_data_labels = X_DATA_LIST, 
                              y_data_label  = "hfe_mu" )
    
    ## 5-FOLD CROSS VALIDATION INPUT
    ## TRAIN GROUPS
    train_groups = TRAINING_GROUPS
#    train_groups = NH2_TRAINING_GROUPS
#    train_groups = CONH2_TRAINING_GROUPS
#    train_groups = OH_TRAINING_GROUPS

    ## CV GROUPS
    validation_groups = TESTING_GROUPS
#    validation_groups = NH2_TESTING_GROUPS
#    validation_groups = CONH2_TESTING_GROUPS
#    validation_groups = OH_TESTING_GROUPS
 
    ## 5-FOLD CROSS VALIDATION WITH MINIMUM FEATURES IDENTIFIED ABOVE
    ## LASSO FEATURES
    lasso_features = []
    lasso_features += [ "theta_48", ]
    lasso_features += [ "theta_90", ]
    lasso_features += [ "num_hbonds_all", ]
    lasso_features += [ "num_hbonds_sam_water", ]
    lasso_features += [ "hbond_sam_water_0", ]
    
    ## LASSO FEATURE INDICES
    lasso_feature_indices = np.array([ True if feature in lasso_features else False
                                       for feature in DATA_LABELS ])
    
    ## MINIMUM FEATURES MASK
    mask        = lasso_feature_indices
    mask_labels = [ ll for ll, mm in zip( DATA_LABELS, mask ) 
                    if mm == True ]
    
    ## CREATE PLACEHOLDERS
    y_orig    = []
    y_predict = []
    reg_coefs = []
    rmse      = []
        
    ## LOOP THROUGH SAMPLES
    for ii in range(n_samples):
        ## REDUCE TO ONLY SAMPLE ONE (FOR NOW)
        X = XX[:,ii,:]
        y = yy[:,ii]
        
        ## RESCALE DATA
        X  = rescale_data( X )
                
        ## GENERATE EMPTY Y PREDICTION ARRAY
        yo = np.empty( shape = ( 0, ) )
        yp = np.empty( shape = ( 0, ) )
        
        ## GENERATE EMPTY COEFS ARRAY
        rc = np.empty( shape = ( 0, len(mask_labels) ) )
        
        for nn in range( len(train_groups) ):        
            ## TRAINING DATA
            X_train = X[train_groups[nn],:][:,mask]
            y_train = y[train_groups[nn]]
            
            ## VALIDATION DATA
            X_val   = X[validation_groups[nn],:][:,mask]
            y_val   = y[validation_groups[nn]]
              
            ## PERFORM LINEAR REGRESSION
            regressor = LinearRegression() # linear regression class
            
            ## TRAIN LINEAR REGRESSION MODEL ON TRAINING PCA DATA
            regressor.fit( X_train, y_train )
            
            ## UPDATE PREDICT Y VALUES
            yp = np.hstack(( yp, regressor.predict( X_val ) ))
            
            ## UPDATE ORIG ARRAY
            yo = np.hstack(( yo, y_val ))
            
            ## UPDATE COEFS ARRAY
            rc = np.vstack(( rc, regressor.coef_ ))
            print( regressor.intercept_ )
                
        ## COMPUTE MSE & RMSE
        MSE  = metrics.mean_squared_error( yo, yp )
        rmse.append( np.sqrt( MSE ) )
        
        ## UPDATE LISTS
        y_orig.append( yo )
        y_predict.append( yp )
        reg_coefs.append( rc )
    
    ## COMPUTE STATISTICS
    y_indus     = np.mean( y_orig, axis = 0 )
    y_indus_err = np.std( y_orig, axis = 0 )
    y_pred      = np.mean( y_predict, axis = 0 )
    y_pred_err  = np.std( y_predict, axis = 0 )
    RMSE        = np.mean( rmse )
    ERR         = np.std( rmse )
    coefs       = np.mean( reg_coefs, axis = 0 )
    coefs_err   = np.std( reg_coefs, axis = 0 )

    ## PLOT PARITY
    fig_path = os.path.join( working_dir, "lasso_parity_minimum_features_5_fold_a_eq_2.0_largeaxis" )
    plot_parity( [ y_indus ],
                 [ y_pred ],
                 xerr         = y_indus_err,
                 yerr         = y_pred_err,
                 title        = r"RMSE = {:.2f}$\pm${:.2f}".format( RMSE, ERR ),
                 xlabel       = r"$\mu_{INDUS}$",
                 ylabel       = r"$\mu_{predicted}$",
                 xticks       = [ 30, 110, 10 ],
                 yticks       = [ 30, 110, 10 ],
                 point_labels = None,
                 fig_path     = fig_path, )
    
    ## MATCH MASK LABELS WITH FEATURE LABELS
    sorted_labels = []
    sorted_coefs = np.zeros_like( coefs )
    sorted_coefs_err = np.zeros_like( coefs_err )
    
    for ii, lf in enumerate(lasso_features):
        for jj, ml in enumerate(mask_labels):
            if lf == ml:
                sorted_labels.append( ml )
                sorted_coefs[:,ii] = coefs[:,jj]
                sorted_coefs_err[:,ii] = coefs_err[:,jj]

    ## PLOT SALIENCY                
#    coefs = reg_coefs / np.abs( coefs ).max()
    fig_path = os.path.join( working_dir, "lasso_saliency_minimum_features_a_eq_2.0_largeaxis" )
    plot_saliency( sorted_coefs.mean( axis = 0 ),
                   yerr      = sorted_coefs_err.mean( axis = 0 ),
                   title     = None,
                   ylabel    = r"Importance",
                   yticks    = [ -16.0, 32.0, 8.0 ],
                   labels    = sorted_labels,
                   fig_path  = fig_path, ) 
    
    fig_path = os.path.join( working_dir, "lasso_saliency_minimum_features_5_fold_a_eq_2.0_largeaxis" )
    plot_multi_saliency( sorted_coefs,
                         yerr      = sorted_coefs_err,
                         title     = None,
                         ylabel    = r"Importance",
                         yticks    = [ -16.0, 32.0, 8.0 ],
                         labels    = sorted_labels,
                         fig_path  = fig_path, )
