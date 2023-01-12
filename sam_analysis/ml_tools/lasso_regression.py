"""
lasso_regression.py
This script contains the lasso regression class

CREATED ON: 12/15/2020

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
from sklearn.linear_model import Lasso
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
## FUNCTIONS
##############################################################################
## LASSO REGRESSION CLASS
class LassoRegression:
    """class object used to perform lasso regression"""
    def __init__( self,
                  alpha     = 2.0,
                  tol       = 1e-5,
                  max_iter  = 5000, ):
        ## INITIALIZE VARIABLES IN CLASS
        self.alpha     = alpha
        self.tol       = tol
        self.max_iter  = max_iter

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

            ## INITIALIZE LASSO
            regressor = Lasso( alpha    = self.alpha, 
                               tol      = self.tol, 
                               max_iter = self.max_iter )

            ## TRAIN LINEAR REGRESSION MODEL ON FULL DATA SET
            regressor.fit( X, y )

            ## UPDATE PREDICT Y TRAIN
            yp = regressor.predict( X )

            ## UPDATE COEFS ARRAY
            w = regressor.coef_

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

    ## COMPUTE LASSO REGRESSION WITH K-FOLD
    def compute_kfold( self, analysis_data, n_samples = None ):
        """ASSUMES DATA HAS BEEN PROPERTLY RESCALED"""
        ## CREATE PLACEHOLDERS
        y_orig          = []
        y_train_orig    = []
        y_test_predict  = []
        y_train_predict = []
        weights         = []
        rmse            = []
        rmse_train      = []

        ## LOOP THROUGH SAMPLES
        for data in analysis_data.values():
            ## SPLIT TO X AND Y DATA 
            X_df = data.drop( "hfe_mu", axis = 1 )
            y_df = data["hfe_mu"]

            ## GENERATE EMPTY Y PREDICTION ARRAY
            yo  = np.empty( shape = ( 0, ) )
            yot = np.empty( shape = ( 0, ) )
            yp  = np.empty( shape = ( 0, ) )
            yt  = np.empty( shape = ( 0, ) )

            ## GENERATE EMPTY COEFS ARRAY
            rc = np.empty( shape = ( 0, X_df.shape[1] ) )

            ## LOOP THROUGH FOLDS
            for nn in range( len(TRAINING_GROUPS) ):
                ## TRAINING DATA
                X_train = X_df.iloc[ TRAINING_GROUPS[nn], : ]
                y_train = y_df.iloc[ TRAINING_GROUPS[nn] ]

                ## TESTING DATA
                X_test = X_df.iloc[ TESTING_GROUPS[nn], : ]
                y_test = y_df.iloc[ TESTING_GROUPS[nn] ]

                ## IF N_SAMPLES IS NOT NONE REDUCE SIZE
                if n_samples is not None:
                    X_train = X_train[:n_samples]
                    y_train = y_train[:n_samples]

                ## INITIALIZE LASSO
                regressor = Lasso( alpha    = self.alpha, 
                                   tol      = self.tol, 
                                   max_iter = self.max_iter )

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
                rc = np.vstack(( rc, regressor.coef_ ))

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
            weights.append( pd.DataFrame( rc, columns = X_df.columns.values ) )
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

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## WORKING DIRECTORY
    project_dir = r"/home/bdallin/python_projects/sam_analysis/sam_analysis"
    data_dir    = "raw_data"

    ## DATA FILE
    data_pkl = r"raw_regression_data.pkl"

    ## FIGURE PATHS
    manuscript_dir = r"/mnt/c/Users/bdallin/Box Sync/univ_of_wisc/manuscripts/chemically_heterogeneous_sams"
    figure_dir     = r"misc_figures"

    ## LOAD DATA
    path_data_pkl = os.path.join( project_dir, data_dir, data_pkl )
    raw_data      = load_pkl( path_data_pkl )

    ## OPTIMAL HYPERPARAMETERS (DETERMINED FROM CV)
    threshold = 0.90 # correlation threshold
    alpha     = 2.5  # regularization constant
    tol       = 1e-5 # lasso tolerance
    max_iter  = 5000 # lasso maximum iteration, if tol not met

    ## INITIALIZE LASSO
    lasso_obj = LassoKFold( threshold = threshold,
                            alpha     = alpha,
                            tol       = tol,
                            max_iter  = max_iter )

    ## COMPUTE 5-FOLD CV USING LASSO
    lasso_obj.compute( raw_data )

    ## STORE RESULTS
    y_indus     = lasso_obj.y_indus
    y_indus_err = lasso_obj.y_indus_err
    y_pred      = lasso_obj.y_pred
    y_pred_err  = lasso_obj.y_pred_err
    rmse        = lasso_obj.rmse
    rmse_err    = lasso_obj.rmse_err
    weights     = lasso_obj.weights
    weights_err = lasso_obj.weights_err

    ## PLOT PARITY
    fig_path = os.path.join( manuscript_dir, figure_dir, "lasso_parity_a_eq_{:.1f}".format( alpha ) )
    plot_parity( [ y_indus, ],
                [ y_pred, ],
                xerr         = [ y_indus_err ],
                yerr         = [ y_pred_err ],
                guideline    = 1,
                title        = r"Lasso Regression ($\lambda = {:.1f}$)".format( alpha ),
                xlabel       = r"$\mu_{INDUS}$",
                ylabel       = r"$\mu_{predicted}$",
                xticks       = [ 30, 110, 10 ],
                yticks       = [ 30, 110, 10 ],
                point_labels = [ r"RMSE = {:.2f}$\pm${:.2f}".format( rmse, rmse_err ) ],
                fig_path     = fig_path, )

    ## PLOT WEIGHTS
    mask             = np.abs( weights ) > 0.
    non_zero_weights = weights[ mask ]
    non_zero_err     = weights_err[ mask ]
    non_zero_labels  = list(weights.index.values[ mask ])

    ## PLOT
    fig_path = os.path.join( working_dir, "lasso_weights_a_eq_{:.1f}".format( alpha ) )
    plot_bar( [ np.arange( 0, len(non_zero_weights), 1 ) ],
            [ non_zero_weights ], 
            yerr     = [ non_zero_err ],
            xlabel   = non_zero_labels,
            ylabel   = r"Importance",
            yticks   = [  -12,  4,  4 ],
            colors   = [ "grey" ],
            fig_path = fig_path )