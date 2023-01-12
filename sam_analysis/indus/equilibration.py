"""
equilibration.py 
script contains functions to determine the equilibration time of indus

CREATED ON: 12/10/2020

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
## IMPORT SHUTIL
import shutil
## IMPORT NUMPY
import numpy as np

## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORT INDUS FUNCTION
from sam_analysis.indus.wham import IndusWham
## IMPORT HFE FUNCTION
from sam_analysis.indus.hydration_fe import HydrationFreeEnergy

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## EQUILIBRATION CLASS
class Equilibration:
    """class object used to compute equilibration time from indus"""
    def __init__( self,
                  sim_working_dir  = None,
                  input_prefix     = None,
                  time_range       = ( 0., 5000., ),
                  time_step        = 1000.,
                  convergence_time = 3000.,
                  recompute_equil  = False,
                  verbose          = True,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.sim_working_dir  = sim_working_dir
        self.input_prefix     = input_prefix
        self.time_range       = time_range
        self.time_step        = time_step
        self.convergence_time = convergence_time
        self.recompute_equil  = recompute_equil
        self.verbose          = verbose
        
        ## REMOVE START AND END FROM KWARGS
        kwargs.pop( "start", None )
        kwargs.pop( "end", None )
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

    def __str__( self ):
        return "HYDRATION FREE ENERGY"

    ## INSTANCE PREPARING EQUIL TIME
    def prep_equil( self ):
        """FUNCTION PREPARING FOR EQUILIBRATION"""
        ## PATH OUTPUT
        self.output_dir = os.path.join( self.sim_working_dir, "output_files" )
        
        ## PATH WHAM
        self.wham_dir = os.path.join( self.sim_working_dir, "wham" )
        
        ## PATH EQUIL
        self.equil_dir = os.path.join( self.wham_dir, "equilibration" )
        
        ## WHAM FILE
        self.wham_file = os.path.join( self.equil_dir, self.input_prefix + "_wham_{}.csv" )
        
        ## REMOVE EQUIL DIRECTORY IF RECOMPUTE
        if self.recompute_equil is True \
           and os.path.exists( self.equil_dir ) is True:
            shutil.rmtree( self.equil_dir )
        
        ## CREATE EQUIL DIRECTORY PATH (IF NOT EXISTS)
        if os.path.exists( self.equil_dir ) is not True:
            os.mkdir( self.equil_dir )
            
    ## INSTANCE DETERMINING EQUIL TIME
    def time( self ):
        """FUNCTION TO PUT WHAM OUTPUT INTO READABLE FORMAT"""
        ## PREPARE OBJECT
        self.prep_equil()
        
        ## PATH TO PKL
        out_name = self.input_prefix + "_equilibration_time.pkl"
        path_pkl = os.path.join( self.output_dir, out_name )
        
        if self.recompute_equil is True or \
           os.path.exists( path_pkl ) is not True:
            ## INITIALIZE INDUS WHAM
            wham = IndusWham( **self.__dict__ )
            
            ## INITIALZE HFE
            hfe = HydrationFreeEnergy( **self.__dict__ )
               
            ## GET END STEP
            end_step = self.time_range[-1] + self.time_step
            
            ## CREATE LISTS WITH START TIMES
            start_time = np.arange( 0., 
                                    end_step - self.convergence_time,
                                    self.time_step )
            
            ## CREATE LIST WITH END TIMES
            end_time = np.arange( self.convergence_time,
                                  end_step,
                                  self.time_step )
            
            ## CREATE PLACE HOLDER
            equil_mu = np.zeros_like( start_time )
            
            ## LOOP THROUGH END TIMES
            for ii, (start, end) in enumerate( zip( start_time, end_time ) ):
                ## UPDATE WHAM FILE
                equil_file = self.wham_file.format( int(start) )
                
                ## PREPARE WHAM INPUTS
                wham_input = [ start, end, equil_file ]
                
                ## RUN WHAM
                # print( wham_input )
                wham.compute( wham_input )
                
                ## COMPUTE HFE
                hfe.recompute_fe   = True
                hfe.hfe( path_wham = equil_file )
                
                ## UPDATE
                equil_mu[ii] = hfe.mu
            
            ## STORE RESULTS
            results = [ start_time, equil_mu ]
            
            ## SAVE PKL
            save_pkl( results, path_pkl )
        else:
            ## LOAD PKL
            results = load_pkl( path_pkl )
        
        ## STORE RESULTS IN CLASS
        self.time  = results[0]
        self.mu    = results[1]
        
## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE CLASS
    equil = Equilibration( **kwargs )                  
        
    ## ANALYZE WHAM
    equil.time()

    ## GATHER RESULTS
    time = equil.time
    mu   = equil.mu
    
    ## RETURN RESULTS
    return { "equil_time" : time,
             "equil_mu"   : mu }
        
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

    ## WHAM KWARGS
    wham_kwargs = { "spring_weak"    : 2.0,
                    "spring_strong"  : 8.5,
                    "tol"            : 0.00001,
                    "temperature"    : 300,
                    "recompute_wham" : True, }
    ## HFE KWARGS
    hfe_kwargs = { "recompute_fe" : True }
    
    ## KWARGS
    kwargs = {}
    kwargs.update( wham_kwargs )
    kwargs.update( hfe_kwargs )
    
    ## INITIALIZE EQUILIBRATION
    equil = Equilibration( sim_working_dir  = path_traj,
                           input_prefix     = input_prefix,
                           time_range       = ( 0., 5000., ),
                           time_step        = 1000.,
                           convergence_time = 3000.,
                           recompute_equil  = True,
                           verbose          = True,
                           **kwargs )
                  
        
    ## ANALYZE WHAM
    equil.time()
    
    ## RESULTS
    equil_time = equil.time
    equil_mu   = equil.mu
    