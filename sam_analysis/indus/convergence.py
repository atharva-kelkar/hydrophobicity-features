"""
convergence.py 
script contains functions to determine the convergence time of indus

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
## CONVERGENCE CLASS
class Convergence:
    """class object used to compute convergence time from indus"""
    def __init__( self,
                  sim_working_dir    = None,
                  input_prefix       = None,
                  time_range         = ( 0., 5000., ),
                  time_step          = 1000.,
                  equilibration_time = 2000.,
                  recompute_converge = False,
                  verbose            = True,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.sim_working_dir    = sim_working_dir
        self.input_prefix       = input_prefix
        self.time_range         = time_range
        self.time_step          = time_step
        self.equilibration_time = equilibration_time
        self.recompute_converge = recompute_converge
        self.verbose            = verbose
        
        ## REMOVE START AND END FROM KWARGS
        kwargs.pop( "start", None )
        kwargs.pop( "end", None )
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

    def __str__( self ):
        return "HYDRATION FREE ENERGY"

    ## INSTANCE PREPARING CONVERGE TIME
    def prep_converge( self ):
        """FUNCTION PREPARING FOR CONVERGENCE"""
        ## PATH OUTPUT
        self.output_dir = os.path.join( self.sim_working_dir, "output_files" )
        
        ## PATH WHAM
        self.wham_dir = os.path.join( self.sim_working_dir, "wham" )
        
        ## PATH CONVERGE
        self.converge_dir = os.path.join( self.wham_dir, "convergence" )
        
        ## WHAM FILE
        self.wham_file = os.path.join( self.converge_dir, self.input_prefix + "_wham_{}.csv" )
        
        ## REMOVE CONVERGE DIRECTORY IF RECOMPUTE
        if self.recompute_converge is True \
           and os.path.exists( self.converge_dir ) is True:
            shutil.rmtree( self.converge_dir )
        
        ## CREATE CONVERGE DIRECTORY PATH (IF NOT EXISTS)
        if os.path.exists( self.converge_dir ) is not True:
            os.mkdir( self.converge_dir )
            
    ## INSTANCE DETERMINING CONVERGE TIME
    def time( self ):
        """FUNCTION TO PUT WHAM OUTPUT INTO READABLE FORMAT"""
        ## PREPARE OBJECT
        self.prep_converge()
        
        ## PATH TO PKL
        out_name = self.input_prefix + "_convergence_time.pkl"
        path_pkl = os.path.join( self.output_dir, out_name )
        
        if self.recompute_converge is True or \
           os.path.exists( path_pkl ) is not True:
            ## INITIALIZE INDUS WHAM
            wham = IndusWham( **self.__dict__ )
            
            ## INITIALZE HFE
            hfe = HydrationFreeEnergy( **self.__dict__ )
                                      
            ## CREATE LIST WITH END TIMES
            end_time = np.arange( self.equilibration_time,
                                  self.time_range[-1],
                                  self.time_step ) + self.time_step
            
            ## CREATE PLACE HOLDER
            start_time = np.zeros_like( end_time )
            equil_mu   = np.zeros_like( end_time )
            
            ## LOOP THROUGH END TIMES
            for ii, end in enumerate( end_time ):
                ## UPDATE WHAM FILE
                start = end - self.equilibration_time
                converge_file = self.wham_file.format( int(start) )
                
                ## PREPARE WHAM INPUTS
                wham_input = [ self.equilibration_time, end, converge_file ]
                
                ## RUN WHAM
                wham.compute( wham_input )
                
                ## COMPUTE HFE
                hfe.recompute_fe   = True
                hfe.hfe( path_wham = converge_file )
                
                ## UPDATE
                start_time[ii] = start 
                equil_mu[ii]   = hfe.mu
            
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
    converge = Convergence( **kwargs )                  
        
    ## ANALYZE WHAM
    converge.time()
    
    ## GATHER RESULTS
    time = converge.time
    mu   = converge.mu
            
    ## RETURN RESULTS
    return { "converge_time" : time,
             "converge_mu"   : mu }
        
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
    converge = Convergence( sim_working_dir    = path_traj,
                            input_prefix       = input_prefix,
                            time_range         = ( 0., 5000., ),
                            time_step          = 1000.,
                            equilibration_time = 2000.,
                            recompute_converge = True,
                            verbose            = True,
                            **kwargs )                 
    
    ## ANALYZE WHAM
    converge.time()
    
    ## RESULTS
    converge_time = converge.time
    converge_mu   = converge.mu
    