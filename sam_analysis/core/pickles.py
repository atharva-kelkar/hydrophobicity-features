"""
pickles.py 
script contains loading and saving pickle functions

CREATED ON: 09/25/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:
"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import pickle

##############################################################################
## FUNCTIONS
##############################################################################
## FUNCTION TO LOAD PICKLE FILES
def load_pkl( path_pkl ):
    r'''
    Function to load data from pickle file
    '''
    ## WRITE OUT
    print( "LOADING PKL FILE...")
    with open( path_pkl, 'rb' ) as input:
        data = pickle.load( input )
    ## WRITE OUT
    print( "LOADED PKL FROM {}\n".format( path_pkl ) )
        
    return data

## FUNCTION TO SAVE PICKLE FILES
def save_pkl( data, path_pkl ):
    r'''
    Function to save data as pickle
    '''
    with open( path_pkl, 'wb' ) as output:
        pickle.dump( data, output, pickle.HIGHEST_PROTOCOL )
    ## WRITE OUT
    print( "PKL SAVED TO {}\n".format( path_pkl ) )

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## PYTHON MODULES
    import os
    import numpy
    
    ## CHECK SERVER PATH
    from sam_analysis.core.check_tools import check_server_path
    ## TESTING DIRECTORY
    test_dir = r"/mnt/r/python_projects/sam_analysis/sam_analysis/testing"
    
    ## SAM DIRECTORY
    sam_dir = r"mixed_polar_sam"
            
    ## TEST PICKLE
    path_pkl = os.path.join(test_dir, sam_dir, "test_pickling.pkl")
    
    ## PATH
    path_pkl = check_server_path( path_pkl )
    
    ## GENERATE RANDOM ARRAY
    rand_array = numpy.random.rand( 50, 50 )
    
    ## TEST SAVE
    save_pkl( rand_array, path_pkl )
    
    ## CHECK SAVE
    print( "PKL SAVED: {}".format( os.path.exists( path_pkl ) ) )
    
    ## TEST LOAD
    load_array = load_pkl( path_pkl )
    
    ## CHECK PKL
    print( "LOADED CORRECTLY: {}".format( numpy.all( load_array == rand_array ) ) )
    
    ## CLEAN UP DIRECTORY
    os.remove( path_pkl )
    