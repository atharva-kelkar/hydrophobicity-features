"""
hydration_fe.py 
script contains functions to calculate the hydration free energy from 
indus wham output

CREATED ON: 12/09/2020

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
## IMPORT NUMPY
import numpy as np

## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## FUNCTION TO LOAD CSV FILES
from sam_analysis.core.misc_tools import load_csv

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## HYDRATION FREE ENERGY CLASS
class HydrationFreeEnergy:
    """class object used to indus wham"""
    def __init__( self,
                  sim_working_dir = None,
                  input_prefix    = None,
                  temperature     = 300.,
                  recompute_fe    = False,
                  verbose         = True,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.sim_working_dir = sim_working_dir
        self.input_prefix    = input_prefix
        self.recompute_fe    = recompute_fe
        self.verbose         = verbose

        ## VARIABLES
        self.output_dir = os.path.join( self.sim_working_dir, "output_files" )
        self.wham_dir   = os.path.join( self.sim_working_dir, "wham" )

        ## CONSTANTS
        kB            = 8.314463e-3      # BOLTZMANN CONSTANT IN KJ/MOL/K
        kT            = kB * temperature
        self.neg_beta = -1. / kT
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

    def __str__( self ):
        return "HYDRATION FREE ENERGY"
        
    ## INSTANCE ANALYZING WHAM OUTPUT
    def hfe( self, path_wham = None ):
        """FUNCTION TO GET HFE FROM WHAM OUTPUT"""
        ## PATH TO PKL
        out_name = self.input_prefix + "_wham.pkl"
        path_pkl = os.path.join( self.output_dir, out_name )

        ## PATH TO WHAM FILE
        if path_wham is None:
            path_wham = os.path.join( self.wham_dir, self.input_prefix + "_wham.csv" )
            
        if self.recompute_fe is True or \
           os.path.exists( path_pkl ) is not True:             
            ## LOAD CSV
            data = load_csv( path_wham )
                        
            ## EXTRACT FREE ENERGY COLUMN
            d_energies = data[:,2]
            
            ## RESHAPE INTO 2D ARRAYS
            m = int( np.sqrt( len(data[:,0]) ) )
            coordinate = np.arange( 0, m, 1 )
            d_energies = np.reshape( d_energies, ( m, m ) )
            
            ## CALCULATE UNNORMALIZE PROBABILITY
            probability = np.exp( self.neg_beta * d_energies )
     
            ## NORMALIZE PROBABILITY ALONG X-AXIS
            normx_probability = probability.sum( axis = 1 ) / probability.sum()
            
            ## CALCULATE FREE ENERGY
            free_energies = -1. * np.log( normx_probability )
            
            ## CALCULATE RELATIVE FREE ENERGY
            norm_free_energies = free_energies - free_energies.min()
            
            ## PRINTING
            print( "  UN-NORMALIZED SUM: {:.3f}".format( probability.sum() ) )
            print( "  NORMALIZED SUM: {:.3f}".format( normx_probability.sum() ) )
            print( "  MIN. FREE ENERGY: {:.3f} kT".format( free_energies.min() ) )
            
            ## STORE RESULTS
            results = { "N"       : coordinate,
                        "prob"    : normx_probability,
                        "fe"      : free_energies,
                        "norm_fe" : norm_free_energies,
                        "mu"      : norm_free_energies[0] }
            
            ## SAVE PKL
            save_pkl( results, path_pkl )
        else:
            ## LOAD PKL
            results = load_pkl( path_pkl )
        
        ## STORE RESULTS
        self.N             = results["N"]
        self.probability   = results["prob"]
        self.free_energies = results["fe"]
        self.norm_fe       = results["norm_fe"]
        self.mu            = results["mu"]
    
## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE CLASS
    indus = HydrationFreeEnergy( **kwargs )
        
    ## ANALYZE WHAM
    indus.hfe()
    
    ## GATHER RESULTS
    n    = indus.N
    prob = indus.probability
    fe   = indus.norm_fe
    mu   = indus.mu
        
    ## RETURN RESULTS
    return { "hfe_N"    : n,
             "hfe_prob" : prob,
             "hfe_dist" : fe,
             "hfe_mu"   : mu }
        
#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## IMPORT CHECK SERVER PATH
    from sam_analysis.core.check_tools import check_server_path
    ## TESTING DIRECTORY
    test_dir = r"/mnt/r/python_projects/sam_analysis/sam_analysis/testing"
    
    ## SAM DIRECTORY
    sam_dir = r"ordered_ch3_sam_indus"
    
    ## WORKING DIR
    working_dir = os.path.join(test_dir, sam_dir)
    
    ## LOAD TRAJECTORY
    path_traj = check_server_path( working_dir )

    ## LOAD TRAJECTORY
    input_prefix = "sam_indus"
    
    ## INITIALIZE HFE
    indus = HydrationFreeEnergy( sim_working_dir = path_traj,
                                 input_prefix    = input_prefix,
                                 recompute_fe    = True,
                                 verbose         = True, )
    
    ## PATH TO WHAM FILE
    path_wham = os.path.join( path_traj, "wham", input_prefix + "_wham.csv" )
    
    ## ANALYZE WHAM
    indus.hfe( path_wham )
    
    ## RESULTS
    N                  = indus.N
    probability        = indus.probability
    free_energies      = indus.free_energies
    norm_free_energies = indus.norm_fe
    mu                 = indus.mu
    
    print( "MU: {}".format( mu ) )
    