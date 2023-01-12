"""
data_tools.py
This script contains tools to prepare or organize ML data

CREATED ON: 10/16/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
### IMPORT MODULES
##############################################################################
import numpy as np
import pandas as pd

##############################################################################
## FUNCTIONS
##############################################################################
## FUNCTION TO RESCALE DATA
def rescale_data( data, columns = None, index = None ):
    r'''
    Rescale data such that mean = 0 and var = 1
    '''
    ## COMPUTE COLUMNWISE MEAN
    mu = data.mean( axis = 0 )

    ## COMPUTE COLUMNWISE STDEV
    stdev = data.std( axis = 0 )
    
    ## RESCALE X DATA
    data_rescaled = np.divide( ( data - mu ), stdev,
                            #    out = np.zeros_like( data ),
                               where = stdev != 0 )
    
    ## RETURN RESULT
    if columns is None and index is None:
        return data_rescaled
    else:
        ## CONVERT TO DF
        return pd.DataFrame( data_rescaled, 
                             index   = index,
                             columns = columns )   
    
## REMOVE CORRELATED FEATURES
def remove_correlated_features( X, y,
                                threshold = 0.9,
                                physical_features = [  "theta_48", "theta_60", "theta_90", "theta_110", "phi_180" ], ):
    ## SET PLACEHOLDER
    features_added = 1

    ## CREATE FOR PLOT
    orig_xy = pd.concat([ X, y ], axis = 1 )
    orig_coefs = np.corrcoef( orig_xy.transpose() )

    ## LOOP THROUGH FEATURES
    while features_added > 0:
        ## COMPUTE PEARSON'S CORRELATION
        xy = pd.concat([ X, y ], axis = 1 )
        corr_coefs = np.corrcoef( xy.transpose() )

        ## CREATE PLACEHOLD FOR FEATURES
        corr_features = set()

        ## COLUMNS
        column_labels = X.columns.values

        for ii in range(len(column_labels)):
            for jj in range(ii):
                ## ELIMINATE CORRELATED FEATURE
                if corr_coefs[ii,jj] >= threshold \
                and column_labels[ii] not in corr_features:
                    ## REMOVE FEATURE LEAST CORRELATED WITH MU, UNLESS PHYSICAL
                    if corr_coefs[ii,-1] < corr_coefs[jj,-1]:
                        if column_labels[ii] not in physical_features:
                            corr_features.add(column_labels[ii])
                        else:
                            corr_features.add(column_labels[jj])
                    else:
                        if column_labels[jj] not in physical_features:
                            corr_features.add(column_labels[jj])
                        else:
                            corr_features.add(column_labels[ii])
        
        ## ELIMINATE CORRELATED DATA
        X.drop( corr_features, axis = 1, inplace = True )

        ## UPDATE FEATURES ADDED
        features_added = len(corr_features)
   
    ## Seaborn plotting object for heatmap
    sns_kwargs = { "data"    : orig_coefs,
                   "xlabels" : orig_xy.columns.values,
                   "ylabels" : orig_xy.columns.values }

    ## RETURN
    return X, sns_kwargs
