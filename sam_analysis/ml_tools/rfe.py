"""
random_forest_regression.py
This script contains the random forest regression class

CREATED ON: 01/05/2021

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT OS
import os

## IMPORT RANDOM
import random

## IMPORT NUMPY
import numpy as np  # Used to do math functions

## IMPORT PANDAS
import pandas as pd

## IMPORT REGRESSION FUNCTIONS FROM SKLEARN
from sklearn.feature_selection import RFE
from sklearn import metrics

## IMPORT GLOBAL INPUTS
from sam_analysis.ml_tools.globals import X_DATA_LIST, RANDOM_SET, \
                                          TRAINING_GROUPS, TESTING_GROUPS

## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl

## IMPORT CHECK SERVER PATH
from sam_analysis.core.check_tools import check_server_path

## IMPORT PLOTTING TOOLS
from sam_analysis.plotting.plots import plot_parity, plot_bar, plot_line

##############################################################################
## CLASSES AND FUNCTIONS
##############################################################################
## RECURSIVE FEATURE SELECTION CLASS
class RecursiveFeatureSelection:
    """class object used to perform random forest regression"""
    def __init__( self,
                  estimator,
                  n_features = 5, ):
        ## INITIALIZE VARIABLES IN CLASS
        self.estimator  = estimator
        self.n_features = n_features

    ## COMPUTE LASSO REGRESSION
    def compute( self, analysis_data, n_samples = None ):
        """ASSUMES DATA HAS BEEN PROPERTLY RESCALED"""
        ## CREATE PLACEHOLDERS
        y_orig_list  = []
        y_pred_list  = []
        weights_list = []
        rmse_list    = []

        ## LOOP THROUGH SAMPLES
        for key, data in analysis_data.items():
            ## SPLIT TO X AND Y DATA 
            X_df = data.drop( "hfe_mu", axis = 1 )
            y_df = data["hfe_mu"]

            ## TRAINING DATA
            X = X_df.iloc[ RANDOM_SET, : ]
            y = y_df.iloc[ RANDOM_SET ]

            ## IF N_SAMPLES IS NOT NONE REDUCE SIZE
            if n_samples is not None:
                X = X[:n_samples]
                y = y[:n_samples]

            ## INITIALIZE RFE
            selector = RFE( self.estimator,
                            n_features_to_select = self.n_features, )

            ## TRAIN LINEAR REGRESSION MODEL ON FULL DATA SET
            selector.fit( X, y )

            ## UPDATE PREDICT Y TRAIN
            yp = selector.predict( X )

            ## UPDATE COEFS ARRAY
            w = selector.support_

            ## COMPUTE MSE & RMSE
            MSE  = metrics.mean_squared_error( y, yp )
            RMSE = np.sqrt( MSE )

            ## UPDATE LISTS
            y_orig_list.append( np.array(y) )
            y_pred_list.append( yp )
            weights_list.append( pd.DataFrame( w, index = X_df.columns.values ) )
            rmse_list.append( RMSE )

        ## STORE FULL RESULTS
        y_orig_list = np.array(y_orig_list).transpose()
        y_pred_list = np.array(y_pred_list).transpose()
        rmse_list   = np.array(rmse_list)
        self.y_orig_df  = pd.DataFrame( y_orig_list, columns = analysis_data.keys() )
        self.y_pred_df  = pd.DataFrame( y_pred_list, columns = analysis_data.keys() )
        self.weights_df = pd.concat([ ww for ww in weights_list ], axis = 1 )
        self.weights_df.columns = analysis_data.keys()
        self.rmse_df    = pd.DataFrame( rmse_list, index = analysis_data.keys(), columns = [ "rmse" ] ).transpose()

        ## COMPUTE STATISTICS
        self.y_indus     = np.mean( self.y_orig_df, axis = 1 )
        self.y_indus_err = np.std( self.y_orig_df, axis = 1 )
        self.y_pred      = np.mean( self.y_pred_df, axis = 1 )
        self.y_pred_err  = np.std( self.y_pred_df, axis = 1 )
        self.rmse        = np.mean( self.rmse_df.loc["rmse"] )
        self.rmse_err    = np.std( self.rmse_df.loc["rmse"] )
        self.weights     = np.mean( self.weights_df, axis = 1 )
        self.weights_err = np.std( self.weights_df, axis = 1 )

    ## COMPUTE RANDOM FOREST REGRESSION WITH K-FOLD
    def compute_kfold( self, raw_data, n_samples = None ):
        ## CREATE PLACEHOLDERS
        y_orig          = []
        y_train_orig    = []
        y_test_predict  = []
        y_train_predict = []
        weights         = []
        rmse            = []
        rmse_train      = []

        ## LOOP THROUGH SAMPLES
        for data in raw_data.values():
            ## SPLIT TO X AND Y DATA 
            X_df = data.drop( "hfe_mu", axis = 1 )
            y_df = data["hfe_mu"]

            ## RESCALE DATA
            X_rescaled_df = rescale_data( X_df,
                                          columns = X_df.columns.values,
                                          index   = X_df.index.values ) # outputs np.array
            y_rescaled_df = rescale_data( y_df,
                                          columns = None,
                                          index   = y_df.index.values ) # outputs np.array

            ## REMOVE ZERO COLUMNS
            non_zero_mask = np.abs( X_rescaled_df.sum( axis = 0 ) ) > 0
            
            ## GET NON-ZERO COLUMNS
            column_labels = X_rescaled_df.columns.values[non_zero_mask]
            
            ## FILTER OUT ZERO COLUMNS
            X_rescaled_df = X_rescaled_df[column_labels]

            ## REMOVE CORRELATED FEATURES
            X_rescaled_df, _ = remove_correlated_features( X_rescaled_df, 
                                                           y_rescaled_df,
                                                           threshold = self.threshold )
            
            ## GENERATE EMPTY Y PREDICTION ARRAY
            yo  = np.empty( shape = ( 0, ) )
            yot = np.empty( shape = ( 0, ) )
            yp  = np.empty( shape = ( 0, ) )
            yt  = np.empty( shape = ( 0, ) )

            ## GENERATE EMPTY COEFS ARRAY
            rc = np.empty( shape = ( 0, X_rescaled_df.shape[1] ) )

            ## LOOP THROUGH FOLDS
            for nn in range( len(TRAINING_GROUPS) ):
                ## TRAINING DATA
                X_train = X_rescaled_df.iloc[ TRAINING_GROUPS[nn], : ]
                y_train = y_df.iloc[ TRAINING_GROUPS[nn] ]

                ## TESTING DATA
                X_test = X_rescaled_df.iloc[ TESTING_GROUPS[nn], : ]
                y_test = y_df.iloc[ TESTING_GROUPS[nn] ]

                ## IF N_SAMPLES IS NOT NONE REDUCE SIZE
                if n_samples is not None:
                    X_train = X_train[:n_samples]
                    y_train = y_train[:n_samples]

                ## INITIALIZE RANDOM FOREST
                regressor = RandomForestRegressor( n_estimators = self.n_trees,
                                                   max_samples  = self.max_samples,
                                                   max_features = self.max_features,
                                                   max_depth    = self.max_depth, )

                ## TRAIN LINEAR REGRESSION MODEL ON TRAINING DATA
                regressor.fit( X_train, y_train )

                ## UPDATE PREDICT Y TRAIN
                yt  = np.hstack(( yt, regressor.predict( X_train ) ))

                ## UPDATE PREDICT Y TEST
                yp  = np.hstack(( yp, regressor.predict( X_test ) ))

                ## UPDATE ORIG ARRAY
                yo = np.hstack(( yo, y_test ))
                yot = np.hstack(( yot, y_train ))

                ## UPDATE COEFS ARRAY
                rc = np.vstack(( rc, regressor.feature_importances_ ))

            ## COMPUTE MSE & RMSE
            MSE        = metrics.mean_squared_error( yo, yp )
            RMSE       = np.sqrt( MSE )
            MSE_TRAIN  = metrics.mean_squared_error( yot, yt )
            RMSE_TRAIN = np.sqrt( MSE_TRAIN )

            ## UPDATE LISTS
            y_orig.append( yo )
            y_train_orig.append( yot )
            y_test_predict.append( yp )
            y_train_predict.append( yt )
            weights.append( pd.DataFrame( rc, columns = X_rescaled_df.columns.values ) )
            rmse.append( RMSE )
            rmse_train.append( RMSE_TRAIN )

        ## COMPUTE STATISTICS
        self.y_indus           = np.mean( y_orig, axis = 0 )
        self.y_indus_err       = np.std( y_orig, axis = 0 )
        self.y_train_indus     = np.mean( y_train_orig, axis = 0 )
        self.y_train_indus_err = np.std( y_train_orig, axis = 0 )
        self.y_pred            = np.mean( y_test_predict, axis = 0 )
        self.y_pred_err        = np.std( y_test_predict, axis = 0 )
        self.y_train_pred      = np.mean( y_train_predict, axis = 0 )
        self.y_train_pred_err  = np.std( y_train_predict, axis = 0 )
        self.rmse              = np.mean( rmse )
        self.rmse_err          = np.std( rmse )
        self.rmse_train        = np.mean( rmse_train )
        self.rmse_train_err    = np.std( rmse_train )
        ## COMBINE WEIGHTS
        tmp         = pd.concat([ ww for ww in weights ])
        coefs       = tmp.groupby( level = 0 ).mean()
        coefs_err   = tmp.groupby( level = 0 ).std()
        ## SET NANS TO ZERO
        coefs[ np.isnan(coefs) ] = 0.
        coefs_err[ np.isnan(coefs_err) ] = 0.
        ## COMPUTE WEIGHT STATISTICS
        self.weights     = np.mean( coefs, axis = 0 )
        self.weights_err = np.std( coefs, axis = 0 )
