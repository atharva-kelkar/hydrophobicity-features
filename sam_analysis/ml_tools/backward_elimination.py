"""
backward_elimination.py
This script contains script to run backward elimination for feature selection

CREATED ON: 10/19/2020

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
## IMPORT RANDOM
import random
## IMPORT NUMPY
import numpy as np  # Used to do math functions
## IMPORT REGRESSION FUNCTIONS FROM SKLEARN
from sklearn.linear_model import LinearRegression
## IMPORT RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
## IMPORT SEQUENTIAL FEATURE SELECTION TOOL
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

## IMPORT GLOBAL INPUTS
from sam_analysis.ml_tools.globals import GROUP_LIST, X_DATA_LIST, DATA_LABELS, \
                                          TRAINING_INDICES, VALIDATION_INDICES, \
                                          TESTING_INDICES
## IMPORT DATA TOOLS
from sam_analysis.ml_tools.data_tools import rescale_data, compile_ml_data,     \
                                             remove_zero_columns
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl
## IMPORT PLOTTING TOOLS
from sam_analysis.ml_tools.plotting import plot_parity, plot_validation,        \
                                           plot_saliency

##############################################################################
## CLASSES AND FUNCTIONS
##############################################################################
## BACKWARD ELIMINATION CLASS
class BackwardElimination:
    """class object used to perform backward elimination"""
    def __init__( self,
                  regressor,
                  k_features = 10,
                  floating   = False,
                  scoring    = "neg_mean_squared_error",
                  n_jobs     = 1, ):
        ## INITIALIZE VARIABLES IN CLASS
        self.regressor  = regressor
        self.k_features = k_features
        self.floating   = floating
        self.scoring    = scoring
        self.n_jobs     = n_jobs
        
    ## COMPUTE BACKWARD ELIMINATION
    def compute( self,
                 X_train, y_train,
                 X_test,  y_test, ):
        ## INITIALIZE SFS CLASS
        sfs = SFS( self.regressor,
                   k_features = self.k_features,
                   forward    = False,
                   floating   = self.floating,
                   scoring    = self.scoring,
                   cv         = False,
                   n_jobs     = self.n_jobs, )
        
        ## TRAIN REGRESSION MODEL ON TRAINING DATA
        sfs.fit( X_train, y_train )
        
        ## RETURN RESULTS
        return { "feature_ndx" : sfs.k_feature_idx_,
                 "score"       : sfs.k_score_ }
        
#        ## PREDICT Y_TRAIN
#        y_train_predict = regressor.predict( X_train )
#        
#        ## PREDICT Y_TEST
#        y_test_predict = regressor.predict( X_test )
#        
#        ## COMPUTE MSE & RMSE OF TRAINING DATA
#        mse_train  = metrics.mean_squared_error( y_train, y_train_predict )
#        rmse_train = np.sqrt( mse_train )
#        
#        ## COMPUTE MSE & RMSE OF TEST DATA
#        mse_test   = metrics.mean_squared_error( y_test, y_test_predict )
#        rmse_test  = np.sqrt( mse_test )
#        
#        ## RETURN RESULTS
#        return { "y_train_predict" : y_train_predict,
#                 "y_test_predict"  : y_test_predict,
#                 "rmse_train"      : rmse_train,
#                 "rmse_test"       : rmse_test,
#                 "importance"      : regressor.feature_importances_,
#                 "predictor"       : regressor.predict }

    ## PERFORM CROSS VALIDATION FOR NUM TREES
    def n_trees_cross_validation( self,
                                  X_train, y_train,
                                  X_test,  y_test, 
                                  n_trees = np.arange( 5, 105, 5 ).astype("int") ):
        ## CROSS VALIDATION FOR LASSO PARAMETERS
        rmse_train  = []
        rmse_test   = []
        n_features  = []
        
        ## LOOP THROUGH NUM TREES
        for num in n_trees:
            ## RESET ALPHA
            self.n_estimators = num
            
            ## COMPUTE LINEAR REGRESSION W/ RANDOM FOREST
            results = self.compute( X_train = X_train,
                                    y_train = y_train,
                                    X_test  = X_test,
                                    y_test  = y_test )
            
            ## APPEND RMSE
            rmse_train.append( results["rmse_train"] )
            rmse_test.append( results["rmse_test"] )
            
            ## APPEND FEATURES
            n_features.append( np.sum(np.abs(results["importance"]) > 0) )
        
        ## CONVERT RMSE TO ARRAY
        rmse_train = np.array(rmse_train)
        rmse_test  = np.array(rmse_test)
        n_features = np.array(n_features)
        
        ## RETURN RESULTS
        return { "n_trees"    : n_trees,
                 "rmse_train" : rmse_train,
                 "rmse_test"  : rmse_test,
                 "n_features" : n_features }
        
    ## PERFORM CROSS VALIDATION FOR MIN SAMPLE SPLIT
    def min_sample_cross_validation( self,
                                     X_train, y_train,
                                     X_test,  y_test, 
                                     min_num_splits = np.arange( 2, 11, 1 ).astype("int") ):
        ## CROSS VALIDATION FOR LASSO PARAMETERS
        rmse_train  = []
        rmse_test   = []
        n_features  = []
        
        ## LOOP THROUGH SPLITS
        for n_split in min_num_splits:
            ## RESET ALPHA
            self.min_samples_split = n_split
            
            ## COMPUTE LINEAR REGRESSION W/ RANDOM FOREST
            results = self.compute( X_train = X_train,
                                    y_train = y_train,
                                    X_test  = X_test,
                                    y_test  = y_test )
            
            ## APPEND RMSE
            rmse_train.append( results["rmse_train"] )
            rmse_test.append( results["rmse_test"] )
            
            ## APPEND FEATURES
            n_features.append( np.sum(np.abs(results["importance"]) > 0) )
        
        ## CONVERT RMSE TO ARRAY
        rmse_train = np.array(rmse_train)
        rmse_test  = np.array(rmse_test)
        n_features = np.array(n_features)
        
        ## RETURN RESULTS
        return { "min_num_splits" : min_num_splits,
                 "rmse_train"     : rmse_train,
                 "rmse_test"      : rmse_test,
                 "n_features"     : n_features }

    ## PERFORM CROSS VALIDATION FOR MIN LEAF SPLIT
    def min_leaf_cross_validation( self,
                                   X_train, y_train,
                                   X_test,  y_test, 
                                   min_num_leaves = np.arange( 1, 11, 1 ).astype("int") ):
        ## CROSS VALIDATION FOR LASSO PARAMETERS
        rmse_train  = []
        rmse_test   = []
        n_features  = []
        
        ## LOOP THROUGH SPLITS
        for n_split in min_num_leaves:
            ## RESET ALPHA
            self.min_samples_leaf = n_split
            
            ## COMPUTE LINEAR REGRESSION W/ RANDOM FOREST
            results = self.compute( X_train = X_train,
                                    y_train = y_train,
                                    X_test  = X_test,
                                    y_test  = y_test )
            
            ## APPEND RMSE
            rmse_train.append( results["rmse_train"] )
            rmse_test.append( results["rmse_test"] )
            
            ## APPEND FEATURES
            n_features.append( np.sum(np.abs(results["importance"]) > 0) )
        
        ## CONVERT RMSE TO ARRAY
        rmse_train = np.array(rmse_train)
        rmse_test  = np.array(rmse_test)
        n_features = np.array(n_features)
        
        ## RETURN RESULTS
        return { "min_num_leaves" : min_num_leaves,
                 "rmse_train"     : rmse_train,
                 "rmse_test"      : rmse_test,
                 "n_features"     : n_features }

    ## PERFORM VALIDATION OF TRAINING DATA SIZE
    def data_size_validation( self,
                              X_train, y_train,
                              X_test,  y_test, 
                              train_size = -1 ):
        ## CROSS VALIDATION FOR TRAIN SET SIZE
        ## USE LENGTH OF TRAINING DATA, IF NOT SPECIFIED
        if train_size < 1:
            train_size = len(y_train)
        
        ## CREATE DATA HOLDERS
        train_sizes  = np.arange( 1, train_size+1, 1 )
        rmse_train  = []
        rmse_test   = []
        
        ## LOOP THROUGH SIZES
        for ii in train_sizes:
            ## COMPUTE LINEAR REGRESSION W/ LASSO
            results = self.compute( X_train = X_train[:ii,:],
                                    y_train = y_train[:ii],
                                    X_test  = X_test,
                                    y_test  = y_test )
            
            ## APPEND RMSE
            rmse_train.append( results["rmse_train"] )
            rmse_test.append( results["rmse_test"] )
            
        ## CONVERT RMSE TO ARRAY
        rmse_train = np.array(rmse_train)
        rmse_test  = np.array(rmse_test)
        
        ## RETURN RESULTS
        return { "train_sizes" : train_sizes,
                 "rmse_train"  : rmse_train,
                 "rmse_test"   : rmse_test   }   

                                           
#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## IMPORT OS
#    import os
    ## MANUSCRIPT DIRECTORY
    manuscript_dir = r"/mnt/c/Users/bdallin/Box Sync/univ_of_wisc/manuscripts"
    
    ## TESTING DIRECTORY
    project_dir = r"chemically_heterogeneous_sams"
    
    ## SAM DIRECTORY
    data_dir = r"simulation_raw_data"
    
    ## WORKING DIR
    working_dir = os.path.join( manuscript_dir, project_dir, data_dir )
        
    ## LOAD DATA
    path_X_pkl = os.path.join( working_dir, "unbiased_regression_data.pkl" )
    X_raw_data = load_pkl( path_X_pkl )
    ## LOAD LABEL DATA
    path_y_pkl = os.path.join( working_dir, "indus_regression_data.pkl" )
    y_raw_data = load_pkl( path_y_pkl )
                      
    ## COMPILE DATA
    X, y = compile_ml_data( group_list    = GROUP_LIST,
                            X_raw         = X_raw_data, 
                            y_raw         = y_raw_data, 
                            X_data_labels = X_DATA_LIST, 
                            y_data_label  = "indus_hydration_fe" )
    
    ## RESCALE DATA
    X = rescale_data( X )
    
    ## REMOVE ZERO COLUMNS
    X, data_labels = remove_zero_columns( X, DATA_LABELS )
    
###############################################################################
### BACKWARD ELIMINATION
###############################################################################
    ## SET RUN TYPE
    run_optimization  = True
    optimized         = False
    random_sets       = False
    
    ## SET FIG TYPE
    ext = ".png"
    
    ## IF RANDOM WANTED
    if random_sets is True:
        ## GET SAM INDICES
        indices = np.arange(len(y)).astype("int")
        ## N-FOLD CROSS VALIDATING LOOP (RANDOMIZE AND SHUFFLE)
        ## SHUFFLE INDICES
        random.shuffle( indices )
        ## TRAIN GROUP
        train_group = indices[:40]
        
        ## VALIDATION GROUP
        validation_group = indices[40:50]
        
        ## TEST GROUP (GROUP MODEL NEVER SEES)
        test_group = indices[50:]
        
    else:
        ## TRAIN GROUP
        train_group = TRAINING_INDICES
        
        ## VALIDATION GROUP
        validation_group = VALIDATION_INDICES
        
        ## TEST GROUP (GROUP MODEL NEVER SEES)
        test_group = TESTING_INDICES

    ## TRAINING DATA
    X_train = X[train_group,:]
    y_train = y[train_group]
    
    ## VALIDATION DATA
    X_val   = X[validation_group,:]
    y_val   = y[validation_group]
    
    ## TEST DATA
    X_test  = X[test_group,:]
    y_test  = y[test_group]
                            
### BACKWARD ELIMINATION
    ## GENERAL WORKFLOW

    if run_optimization is True:
#        ## INITIALIZE RANDOM FOREST CLASS
#        regressor = RandomForestRegressor( n_estimators       = 100,
#                                           max_depth          = None,
#                                           min_samples_split  = 4,
#                                           min_samples_leaf   = 2, )
        
        ## INITIALIZE LINEAR REGRESSION CLASS
        regressor = LinearRegression()
        
        ## INITIALIZE BACKWARD ELIMINATION CLASS
        back_elim = BackwardElimination( regressor,
                                         k_features = 10,
                                         floating   = True,
                                         scoring    = "neg_mean_squared_error",
                                         n_jobs     = 1, )
        ## COMPUTE
        results = back_elim.compute( X_train = X_train,
                                     y_train = y_train,
                                     X_test  = X_val,
                                     y_test  = y_val, )
        features = [ ll for ii, ll in enumerate(data_labels) if ii in results["feature_ndx"] ]
        print(features)
#        ## DETERMINE OPTIMAL NUMBER OF TREES
#        n_trees = np.arange( 1, 101, 1 ).astype("int") 
#        rf_results_n_trees = rf_reg.n_trees_cross_validation( X_train = X_train,
#                                                              y_train = y_train,
#                                                              X_test  = X_val,
#                                                              y_test  = y_val,
#                                                              n_trees = n_trees )
#        ## PLOT
#        fig_path = None # os.path.join( working_dir, "rf_num_trees_validation" + ext )
#        plot_validation( x           = rf_results_n_trees["n_trees"], 
#                         y           = [ rf_results_n_trees["rmse_train"], 
#                                         rf_results_n_trees["rmse_test"] ],
#                         yerr        = [],
#                         xlabel      = r"Num. Trees",
#                         ylabel      = r"RMSE (kT)",
#                         xticks      = [ 0, 100, 10 ],
#                         yticks      = [ 0, 16, 2 ],
#                         line_labels = [ "Train", "Test" ],
#                         fig_path    = fig_path )
#    
#        ## PLOT
#        fig_path = os.path.join( working_dir, "rf_num_trees_num_features_validation" + ext )
#        plot_validation( x        = rf_results_n_trees["n_trees"], 
#                         y        = rf_results_n_trees["n_features"],
#                         yerr     = [],
#                         xlabel   = r"Num. Trees",
#                         ylabel   = r"# features",
#                         xticks   = [ 0, 101, 10 ],
#                         yticks   = [ 0, 120, 10 ],
#                         fig_path = fig_path )
#    
#        ## INITIALIZE RANDOM FOREST CLASS
#        rf_reg = RandomForestRegression( n_trees           = 100,
#                                         min_samples_split = 4,
#                                         min_samples_leaf  = 2 )
#                
#        ## DETERMINE OPTIMAL NUMBER IN SAMPLE SPLIT
#        min_num_splits = np.arange( 2, 21, 1 ).astype("int")
#        rf_results_n_trees = rf_reg.min_sample_cross_validation( X_train        = X_train,
#                                                                 y_train        = y_train,
#                                                                 X_test         = X_val,
#                                                                 y_test         = y_val, 
#                                                                 min_num_splits = min_num_splits )
#        ## PLOT
#        fig_path = os.path.join( working_dir, "rf_num_splits_validation" + ext )
#        plot_validation( x           = rf_results_n_trees["min_num_splits"], 
#                         y           = [ rf_results_n_trees["rmse_train"], 
#                                         rf_results_n_trees["rmse_test"] ],
#                         yerr        = [],
#                         xlabel      = r"Min. samples to split",
#                         ylabel      = r"RMSE (kT)",
#                         xticks      = [ 0, 20, 2 ],
#                         yticks      = [ 0, 16, 2 ],
#                         line_labels = [ "Train", "Test" ],
#                         fig_path    = fig_path )
#    
#        ## PLOT
#        fig_path = os.path.join( working_dir, "rf_num_splits_num_features_validation" + ext )
#        plot_validation( x        = rf_results_n_trees["min_num_splits"], 
#                         y        = rf_results_n_trees["n_features"],
#                         yerr     = [],
#                         xlabel   = r"Min. samples to split",
#                         ylabel   = r"# features",
#                         xticks   = [ 0, 20, 2 ],
#                         yticks   = [ 0, 120, 10 ],
#                         fig_path = fig_path )
#        
#        ## INITIALIZE RANDOM FOREST CLASS
#        rf_reg = RandomForestRegression( n_trees           = 100,
#                                         min_samples_split = 4,
#                                         min_samples_leaf  = 2, )
#                
#        ## DETERMINE OPTIMAL NUMBER IN LEAF SPLIT
#        min_num_leaves = np.arange( 1, 21, 1 ).astype("int")
#        rf_results_n_trees = rf_reg.min_leaf_cross_validation( X_train        = X_train,
#                                                               y_train        = y_train,
#                                                               X_test         = X_val,
#                                                               y_test         = y_val, 
#                                                               min_num_leaves = min_num_leaves )
#        ## PLOT
#        fig_path = os.path.join( working_dir, "rf_num_splits_validation" + ext )
#        plot_validation( x           = rf_results_n_trees["min_num_leaves"], 
#                         y           = [ rf_results_n_trees["rmse_train"], 
#                                         rf_results_n_trees["rmse_test"] ],
#                         yerr        = [],
#                         xlabel      = r"Min. leaves to split",
#                         ylabel      = r"RMSE (kT)",
#                         xticks      = [ 0, 20, 2 ],
#                         yticks      = [ 0, 20, 2 ],
#                         line_labels = [ "Train", "Test" ],
#                         fig_path    = fig_path )
#    
#        ## PLOT
#        fig_path = os.path.join( working_dir, "rf_num_splits_num_features_validation" + ext )
#        plot_validation( x        = rf_results_n_trees["min_num_leaves"], 
#                         y        = rf_results_n_trees["n_features"],
#                         yerr     = [],
#                         xlabel   = r"Min. leaves to split",
#                         ylabel   = r"# features",
#                         xticks   = [ 0, 20, 2 ],
#                         yticks   = [ 0, 120, 10 ],
#                         fig_path = fig_path )
#    
#        ## ESTIMATE GENERALIZATION ERROR USING TEST DATA
#        ## OPTIMAL PARAMETERS NUM TREES = 100, NUM SAMPLE SPLIT = 4, NUM LEAF SPLIT = 2
#        ## NUM TREES HAS LITTLE EFFECT ON RMSE, NOT SURE WHY?
#    
#    if optimized is True:
#        ## OPTIMIZED PARAMETERS
#        n_trees           = 100
#        min_samples_split = 4
#        min_samples_leaf  = 2 
#        ## INITIALIZE RANDOM FOREST CLASS
#        rf_reg = RandomForestRegression( n_trees           = n_trees,
#                                         min_samples_split = min_samples_split,
#                                         min_samples_leaf  = min_samples_leaf )
#        
#        ## FIT REGRESSION ON TRAIN AND TEST SETS
#        rf_results = rf_reg.compute( X_train = X_train,
#                                     y_train = y_train,
#                                     X_test  = X_val,
#                                     y_test  = y_val, )
#        
#        ## EXTRACT RESULTS
#        rf_predict      = rf_results["predictor"]
#        importance      = rf_results["importance"]
#        y_train_predict = rf_results["y_train_predict"]
#        rmse_train      = rf_results["rmse_train"]
#        y_val_predict   = rf_results["y_test_predict"]
#        rmse_val        = rf_results["rmse_test"]
#        
#        ## REGRESS TEST SET
#        y_test_predict = rf_predict( X_test )
#        
#        ## COMPUTE MSE & RMSE OF TEST DATA
#        mse_test   = metrics.mean_squared_error( y_test, y_test_predict )
#        rmse_test  = np.sqrt( mse_test )
#        
#        ## PLOT PARITY
#        fig_path = os.path.join( working_dir, "rf_parity" + ext )
#        plot_parity( [ y_val, y_test ], # y_train,
#                     [ y_val_predict, y_test_predict ], #  y_train_predict,
#                     xerr         = [],
#                     yerr         = [],
#                     title        = r"Random Forest Regression",
#                     xlabel       = r"$\mu_{INDUS}$",
#                     ylabel       = r"$\mu_{predicted}$",
#                     xticks       = [ 30, 110, 10 ],
#                     yticks       = [ 30, 110, 10 ],
#                     point_labels = [ r"Valid. RMSE = {:.2f}".format( rmse_val ), 
#                                      r"Test   RMSE = {:.2f}".format( rmse_test ), ],
#                                    # r"Train  RMSE = {:.2f}".format( rmse_train ),
#                     fig_path     = fig_path, )
#    
#        ## PLOT SALIENCY
#        mask                = np.abs( importance ) > 0.
#        non_zero_importance = importance[ mask ]
#        non_zero_importance = non_zero_importance / np.abs( non_zero_importance ).max()
#        non_zero_labels     = [ ll for ll, cc in zip( data_labels, importance ) 
#                                 if np.abs( cc ) > 0 ]
#        
#        fig_path = os.path.join( working_dir, "rf_saliency" + ext )
#        plot_saliency( non_zero_importance,
#                       yerr      = [],
#                       title     = r"Random Forest Regression",
#                       ylabel    = r"Importance",
#                       yticks    = [ -1.0, 1.0, 0.4 ],
#                       labels    = non_zero_labels,
#                       fig_path  = fig_path, )  