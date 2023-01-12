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
from sklearn.ensemble import RandomForestRegressor
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
## RANDOM FOREST REGRESSION CLASS
class RandomForestRegression:
    """class object used to perform random forest regression"""
    def __init__( self,
                  n_trees      = 100,
                  max_samples  = None,
                  max_features = "auto",
                  max_depth    = None,
                  n_features   = 5,
                  random_state = 0,
                  n_procs      = 4 ):
        ## INITIALIZE VARIABLES IN CLASS
        self.n_trees      = n_trees
        self.max_samples  = max_samples
        self.max_features = max_features
        self.max_depth    = max_depth
        self.n_features   = n_features
        self.random_state = random_state
        self.n_procs      = n_procs

    ## COMPUTE LASSO REGRESSION
    def compute( self, analysis_data, n_samples = None ):
        """ASSUMES DATA HAS BEEN PROPERLY RESCALED"""
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

            ## INITIALIZE RANDOM FOREST
            regressor = RandomForestRegressor( n_estimators = self.n_trees,
                                               max_samples  = self.max_samples,
                                               max_features = self.max_features,
                                               max_depth    = self.max_depth,
                                               random_state = self.random_state,
                                               n_jobs       = self.n_procs )
            
            # ## CHECK
            # regressor.fit( X, y )
            # yp = regressor.predict( X )

            ## APPLY RECURSIVE FEATURE ELIMINATION TO FIND FEATURES
            selector = RFE( regressor,
                            n_features_to_select = self.n_features, )

            ## TRAIN LINEAR REGRESSION MODEL ON FULL DATA SET
            selector.fit( X, y )

            ## UPDATE PREDICT Y TRAIN
            yp = selector.predict( X )

            ## STORE REGRESSOR
            regressor = selector.estimator_

            ## UPDATE COEFS ARRAY
            w  = regressor.feature_importances_
            # sorted_idx = w.argsort()
            # w = w[sorted_idx][-6:]
            # wl = X_df.columns.values[sorted_idx][-6:]
            wl = X_df.columns.values[selector.support_]

            ## COMPUTE MSE & RMSE
            MSE  = metrics.mean_squared_error( y, yp )
            RMSE = np.sqrt( MSE )

            ## UPDATE LISTS
            y_orig_list.append( np.array(y) )
            y_pred_list.append( yp )
            weights_list.append( pd.DataFrame( w, index = wl ) )
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

#     ## PERFORM CROSS VALIDATION FOR NUM TREES
#     def n_trees_cross_validation( self,
#                                   X_train, y_train,
#                                   X_test,  y_test, 
#                                   n_trees = np.arange( 5, 105, 5 ).astype("int") ):
#         ## CROSS VALIDATION FOR RANDOM FOREST PARAMETERS
#         rmse_train  = []
#         rmse_test   = []
#         n_features  = []
        
#         ## LOOP THROUGH NUM TREES
#         for num in n_trees:
#             ## RESET NUM ESTIMATORS
#             self.n_estimators = num
            
#             ## COMPUTE LINEAR REGRESSION W/ RANDOM FOREST
#             results = self.compute( X_train = X_train,
#                                     y_train = y_train,
#                                     X_test  = X_test,
#                                     y_test  = y_test )
            
#             ## APPEND RMSE
#             rmse_train.append( results["rmse_train"] )
#             rmse_test.append( results["rmse_test"] )
            
#             ## APPEND FEATURES
#             n_features.append( np.sum(np.abs(results["importance"]) > 0) )
        
#         ## CONVERT RMSE TO ARRAY
#         rmse_train = np.array(rmse_train)
#         rmse_test  = np.array(rmse_test)
#         n_features = np.array(n_features)
        
#         ## RETURN RESULTS
#         return { "n_trees"    : n_trees,
#                  "rmse_train" : rmse_train,
#                  "rmse_test"  : rmse_test,
#                  "n_features" : n_features }
        
#     ## PERFORM CROSS VALIDATION FOR MIN SAMPLE SPLIT
#     def min_sample_cross_validation( self,
#                                      X_train, y_train,
#                                      X_test,  y_test, 
#                                      min_num_splits = np.arange( 2, 11, 1 ).astype("int") ):
#         ## CROSS VALIDATION FOR LASSO PARAMETERS
#         rmse_train  = []
#         rmse_test   = []
#         n_features  = []
        
#         ## LOOP THROUGH SPLITS
#         for n_split in min_num_splits:
#             ## RESET SPLIT
#             self.min_samples_split = n_split
            
#             ## COMPUTE REGRESSION W/ RANDOM FOREST
#             results = self.compute( X_train = X_train,
#                                     y_train = y_train,
#                                     X_test  = X_test,
#                                     y_test  = y_test )
            
#             ## APPEND RMSE
#             rmse_train.append( results["rmse_train"] )
#             rmse_test.append( results["rmse_test"] )
            
#             ## APPEND FEATURES
#             n_features.append( np.sum(np.abs(results["importance"]) > 0) )
        
#         ## CONVERT RMSE TO ARRAY
#         rmse_train = np.array(rmse_train)
#         rmse_test  = np.array(rmse_test)
#         n_features = np.array(n_features)
        
#         ## RETURN RESULTS
#         return { "min_num_splits" : min_num_splits,
#                  "rmse_train"     : rmse_train,
#                  "rmse_test"      : rmse_test,
#                  "n_features"     : n_features }

#     ## PERFORM CROSS VALIDATION FOR MIN LEAF SPLIT
#     def min_leaf_cross_validation( self,
#                                    X_train, y_train,
#                                    X_test,  y_test, 
#                                    min_num_leaves = np.arange( 1, 11, 1 ).astype("int") ):
#         ## CROSS VALIDATION FOR RANDOM FOREST PARAMETERS
#         rmse_train  = []
#         rmse_test   = []
#         n_features  = []
        
#         ## LOOP THROUGH SPLITS
#         for n_split in min_num_leaves:
#             ## RESET ALPHA
#             self.min_samples_leaf = n_split
            
#             ## COMPUTE REGRESSION W/ RANDOM FOREST
#             results = self.compute( X_train = X_train,
#                                     y_train = y_train,
#                                     X_test  = X_test,
#                                     y_test  = y_test )
            
#             ## APPEND RMSE
#             rmse_train.append( results["rmse_train"] )
#             rmse_test.append( results["rmse_test"] )
            
#             ## APPEND FEATURES
#             n_features.append( np.sum(np.abs(results["importance"]) > 0) )
        
#         ## CONVERT RMSE TO ARRAY
#         rmse_train = np.array(rmse_train)
#         rmse_test  = np.array(rmse_test)
#         n_features = np.array(n_features)
        
#         ## RETURN RESULTS
#         return { "min_num_leaves" : min_num_leaves,
#                  "rmse_train"     : rmse_train,
#                  "rmse_test"      : rmse_test,
#                  "n_features"     : n_features }

#     ## PERFORM VALIDATION OF TRAINING DATA SIZE
#     def data_size_validation( self,
#                               X_train, y_train,
#                               X_test,  y_test, 
#                               train_size = -1 ):
#         ## CROSS VALIDATION FOR TRAIN SET SIZE
#         ## USE LENGTH OF TRAINING DATA, IF NOT SPECIFIED
#         if train_size < 1:
#             train_size = len(y_train)
        
#         ## CREATE DATA HOLDERS
#         train_sizes  = np.arange( 1, train_size+1, 1 )
#         rmse_train  = []
#         rmse_test   = []
        
#         ## LOOP THROUGH SIZES
#         for ii in train_sizes:
#             ## COMPUTE REGRESSION W/ RANDOM FOREST
#             results = self.compute( X_train = X_train[:ii,:],
#                                     y_train = y_train[:ii],
#                                     X_test  = X_test,
#                                     y_test  = y_test )
            
#             ## APPEND RMSE
#             rmse_train.append( results["rmse_train"] )
#             rmse_test.append( results["rmse_test"] )
            
#         ## CONVERT RMSE TO ARRAY
#         rmse_train = np.array(rmse_train)
#         rmse_test  = np.array(rmse_test)
        
#         ## RETURN RESULTS
#         return { "train_sizes" : train_sizes,
#                  "rmse_train"  : rmse_train,
#                  "rmse_test"   : rmse_test   }   

# #%%
# ##############################################################################
# ## MAIN SCRIPT
# ##############################################################################
# if __name__ == "__main__":
#     ## SET RUN TYPE
#     run_optimization  = False
#     optimized         = True
#     random_sets       = False
#     remove_correlated = True

#     ## OPTIMIZED PARAMETERS
#     n_trees           = 500
#     min_samples_split = 4
#     min_samples_leaf  = 2 
#     threshold         = 0.85 # correlation coef threshold
    
#     ## WORKING DIRECTORY
#     working_dir = r"/mnt/r/python_projects/sam_analysis/sam_analysis/raw_data"
#     working_dir = check_server_path( working_dir )
    
#     ## FEATURES FILE
#     feature_pkl = r"unbiased_regression_data.pkl"
    
#     ## LABELS FILE
#     label_pkl = r"indus_regression_data.pkl"
    
#     ## LOAD DATA
#     path_X_pkl = os.path.join( working_dir, feature_pkl )
#     X_raw_data = load_pkl( path_X_pkl )
    
#     ## LOAD LABEL DATA
#     path_y_pkl = os.path.join( working_dir, label_pkl )
#     y_raw_data = load_pkl( path_y_pkl )
                      
#     ## COMPILE DATA
#     X, y = compile_ml_data( group_list    = GROUP_LIST,
#                             X_raw         = X_raw_data, 
#                             y_raw         = y_raw_data, 
#                             X_data_labels = X_DATA_LIST, 
#                             y_data_label  = "hfe_mu" )
    
#     ## REDUCE TO ONLY SAMPLE ONE (FOR NOW)
#     X = X[:,0,:]
#     y = y[:,0]
    
#     ## RESCALE DATA
#     X  = rescale_data( X )

#     ## REMOVE ZERO COLUMNS
#     X, data_labels = remove_zero_columns( X, DATA_LABELS )
    
#     ## REMOVE CORRELATED FEATURES
#     if remove_correlated is True:
#         X, data_labels = remove_correlated_features( X, rescale_data(y),
#                                                      data_labels,
#                                                      threshold = threshold, )
# #                                                     physical_features = PHYSICAL_FEATURES, )
# #                                                     plot_corr_matrix = True )

# ###############################################################################
# ### RANDOM FOREST REGRESSION
# ###############################################################################   
#     ## IF RANDOM WANTED
#     if random_sets is True:
#         ## GET SAM INDICES
#         indices = np.arange(len(y)).astype("int")
#         ## N-FOLD CROSS VALIDATING LOOP (RANDOMIZE AND SHUFFLE)
#         ## SHUFFLE INDICES
#         random.shuffle( indices )
#         ## TRAIN GROUP
#         train_group = indices[:48]
        
#         ## TEST GROUP
#         test_group = indices[48:]        
#     else:
#         ## TRAIN GROUP
#         train_group = TRAINING_INDICES
                
#         ## TEST GROUP
#         test_group = TESTING_INDICES

#     ## TRAINING DATA
#     X_train = X[train_group,:]
#     y_train = y[train_group]
        
#     ## TEST DATA
#     X_test  = X[test_group,:]
#     y_test  = y[test_group]
                            
# ### RANDOM FOREST REGRESSION
#     ## GENERAL WORKFLOW
#     ## 1- RANDOMLY SELECT TRAINING, VALIDATION, AND TEST SETS (DONE IN OTHER SCRIPT)
#     ## 2- TRAIN RANDOM FOREST REGRESSION OPTIMIZING NUM TREES, SPLITS, LEAFS WITH TRAINING DATA
#     ## 3- ESTIMATE GENERALIZATION ERROR BY APPLYING TO REGRESSION TO TEST SET
    
#     if run_optimization is True:
#         ## INITIALIZE RANDOM FOREST CLASS
#         rf_reg = RandomForestRegression( n_trees           = 100,
#                                          min_samples_split = 4,
#                                          min_samples_leaf  = 2 )
                
#         ## DETERMINE OPTIMAL NUMBER OF TREES
#         n_trees = np.arange( 1, 1001, 10 ).astype("int") 
#         rf_results_n_trees = rf_reg.n_trees_cross_validation( X_train = X_train,
#                                                               y_train = y_train,
#                                                               X_test  = X_test,
#                                                               y_test  = y_test,
#                                                               n_trees = n_trees )
#         ## PLOT
#         fig_path = None # os.path.join( working_dir, "rf_num_trees_validation" + ext )
#         plot_validation( x           = rf_results_n_trees["n_trees"], 
#                          y           = [ rf_results_n_trees["rmse_train"], 
#                                          rf_results_n_trees["rmse_test"] ],
#                          yerr        = [],
#                          xlabel      = r"Num. Trees",
#                          ylabel      = r"RMSE (kT)",
#                          xticks      = [ 0, 1000, 100 ],
#                          yticks      = [ 0, 16, 2 ],
#                          line_labels = [ "Train", "Test" ],
#                          fig_path    = fig_path )
    
#         ## PLOT
#         fig_path = os.path.join( working_dir, "rf_num_trees_num_features_validation" )
#         plot_validation( x        = rf_results_n_trees["n_trees"], 
#                          y        = rf_results_n_trees["n_features"],
#                          yerr     = [],
#                          xlabel   = r"Num. Trees",
#                          ylabel   = r"# features",
#                          xticks   = [ 0, 1000, 100 ],
#                          yticks   = [ 0, 120, 10 ],
#                          fig_path = fig_path )
    
#         ## INITIALIZE RANDOM FOREST CLASS
#         rf_reg = RandomForestRegression( n_trees           = 100,
#                                          min_samples_split = 4,
#                                          min_samples_leaf  = 2 )
                
#         ## DETERMINE OPTIMAL NUMBER IN SAMPLE SPLIT
#         min_num_splits = np.arange( 2, 21, 1 ).astype("int")
#         rf_results_n_trees = rf_reg.min_sample_cross_validation( X_train        = X_train,
#                                                                  y_train        = y_train,
#                                                                  X_test         = X_test,
#                                                                  y_test         = y_test, 
#                                                                  min_num_splits = min_num_splits )
#         ## PLOT
#         fig_path = os.path.join( working_dir, "rf_num_splits_validation" )
#         plot_validation( x           = rf_results_n_trees["min_num_splits"], 
#                          y           = [ rf_results_n_trees["rmse_train"], 
#                                          rf_results_n_trees["rmse_test"] ],
#                          yerr        = [],
#                          xlabel      = r"Min. samples to split",
#                          ylabel      = r"RMSE (kT)",
#                          xticks      = [ 0, 20, 2 ],
#                          yticks      = [ 0, 16, 2 ],
#                          line_labels = [ "Train", "Test" ],
#                          fig_path    = fig_path )
    
#         ## PLOT
#         fig_path = os.path.join( working_dir, "rf_num_splits_num_features_validation" )
#         plot_validation( x        = rf_results_n_trees["min_num_splits"], 
#                          y        = rf_results_n_trees["n_features"],
#                          yerr     = [],
#                          xlabel   = r"Min. samples to split",
#                          ylabel   = r"# features",
#                          xticks   = [ 0, 20, 2 ],
#                          yticks   = [ 0, 120, 10 ],
#                          fig_path = fig_path )
        
#         ## INITIALIZE RANDOM FOREST CLASS
#         rf_reg = RandomForestRegression( n_trees           = 100,
#                                          min_samples_split = 4,
#                                          min_samples_leaf  = 2, )
                
#         ## DETERMINE OPTIMAL NUMBER IN LEAF SPLIT
#         min_num_leaves = np.arange( 1, 21, 1 ).astype("int")
#         rf_results_n_trees = rf_reg.min_leaf_cross_validation( X_train        = X_train,
#                                                                y_train        = y_train,
#                                                                X_test         = X_test,
#                                                                y_test         = y_test,
#                                                                min_num_leaves = min_num_leaves )
#         ## PLOT
#         fig_path = os.path.join( working_dir, "rf_num_splits_validation" )
#         plot_validation( x           = rf_results_n_trees["min_num_leaves"], 
#                          y           = [ rf_results_n_trees["rmse_train"], 
#                                          rf_results_n_trees["rmse_test"] ],
#                          yerr        = [],
#                          xlabel      = r"Min. leaves to split",
#                          ylabel      = r"RMSE (kT)",
#                          xticks      = [ 0, 20, 2 ],
#                          yticks      = [ 0, 20, 2 ],
#                          line_labels = [ "Train", "Test" ],
#                          fig_path    = fig_path )
    
#         ## PLOT
#         fig_path = os.path.join( working_dir, "rf_num_splits_num_features_validation" )
#         plot_validation( x        = rf_results_n_trees["min_num_leaves"], 
#                          y        = rf_results_n_trees["n_features"],
#                          yerr     = [],
#                          xlabel   = r"Min. leaves to split",
#                          ylabel   = r"# features",
#                          xticks   = [ 0, 20, 2 ],
#                          yticks   = [ 0, 120, 10 ],
#                          fig_path = fig_path )
    
#         ## ESTIMATE GENERALIZATION ERROR USING TEST DATA
#         ## OPTIMAL PARAMETERS NUM TREES = 500, NUM SAMPLE SPLIT = 4, NUM LEAF SPLIT = 2
#         ## NUM TREES HAS LITTLE EFFECT ON RMSE, NOT SURE WHY?
    
#     if optimized is True:
#         ## INITIALIZE RANDOM FOREST CLASS
#         rf_reg = RandomForestRegression( n_trees           = n_trees,
#                                          min_samples_split = min_samples_split,
#                                          min_samples_leaf  = min_samples_leaf )
        
#         ## FIT REGRESSION ON TRAIN AND TEST SETS
#         rf_results = rf_reg.compute( X_train = X_train,
#                                      y_train = y_train,
#                                      X_test  = X_test,
#                                      y_test  = y_test, )
        
#         ## EXTRACT RESULTS
#         rf_predict      = rf_results["predictor"]
#         importance      = rf_results["importance"]
#         y_train_predict = rf_results["y_train_predict"]
#         rmse_train      = rf_results["rmse_train"]
#         y_test_predict   = rf_results["y_test_predict"]
#         rmse_test        = rf_results["rmse_test"]
                
#         ## PLOT PARITY
#         fig_path = os.path.join( working_dir, "rf_parity" )
#         plot_parity( [ y_train, y_test ], # y_train,
#                      [ y_train_predict, y_test_predict ],
#                      xerr         = [],
#                      yerr         = [],
#                      title        = r"Random Forest Regression",
#                      xlabel       = r"$\mu_{INDUS}$",
#                      ylabel       = r"$\mu_{predicted}$",
#                      xticks       = [ 30, 110, 10 ],
#                      yticks       = [ 30, 110, 10 ],
#                      point_labels = [ r"Train RMSE = {:.2f}".format( rmse_train ), 
#                                       r"Test  RMSE = {:.2f}".format( rmse_test ), ],
#                      fig_path     = fig_path, )
    
#         ## PLOT SALIENCY
#         tol                 = 0.03
#         mask                = np.abs( importance ) > tol
#         non_zero_importance = importance[ mask ]
# #        non_zero_importance = non_zero_importance / np.abs( non_zero_importance ).max()
#         non_zero_labels     = [ ll for ll, cc in zip( data_labels, importance ) 
#                                  if np.abs( cc ) > tol ]
        
#         fig_path = os.path.join( working_dir, "rf_saliency" )
#         plot_saliency( non_zero_importance,
#                        yerr      = [],
#                        title     = r"Random Forest Regression",
#                        ylabel    = r"Importance",
#                        yticks    = [ 0.0, 0.7, 0.1 ],
#                        labels    = non_zero_labels,
#                        fig_path  = fig_path, )    
