"""
wham.py 
script contains functions to perform WHAM on a MD INDUS trajectories.
NOTE: this script relies on Grossfield WHAM (v. 2.0.10.2)
Download and installation instructions can be found here.
http://membrane.urmc.rochester.edu/?page_id=126

CREATED ON: 12/08/2020

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
## IMPORT SUBPROCESS
import subprocess

## IMPORT CHECK SERVER PATH
from sam_analysis.core.check_tools import check_server_path
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## FUNCTION TO READ AND WRITE CSV FILES
from sam_analysis.core.misc_tools import load_csv, write_csv

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## WHAM EXECUTABLE (INSTALL THE GROSSMAN WHAM WITH KJ/MOL)
INDUS_PATH = os.path.dirname( os.path.abspath(__file__) ) ## ASSUMES THIS FILE NEVER LEAVE INDUS DIR
WHAM_PATH = "wham/wham-2d"
WHAM_EXE  = "wham-2d"
WHAM      = check_server_path( os.path.join( INDUS_PATH, WHAM_PATH, WHAM_EXE ) )

## CLASS TO INDUS WHAM
class IndusWham:
    """class object used to indus wham"""
    def __init__( self,
                  sim_working_dir = None,
                  input_prefix    = None,
                  start           = 2000,
                  end             = -1,
                  spring_weak     = 2.0,  # WEAK SPRING CONST IN KJ/MOL/N2
                  spring_strong   = 8.5,  # STRONG SPRING CONST IN KJ/MOL/N2
                  tol             = 0.00001,
                  temperature     = 300,
                  recompute_wham  = False,
                  verbose         = True,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.sim_working_dir = sim_working_dir
        self.input_prefix    = input_prefix
        self.spring_weak     = spring_weak
        self.spring_strong   = spring_strong
        self.tol             = tol
        self.temperature     = temperature
        self.recompute_wham  = recompute_wham
        self.verbose         = verbose
        
        ## WHAM INPUT
        self.wham_input = [ start, end, None ]
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

        ## PREPARE WHAM DIRECTORIES
        self.prep_wham()

    def __str__( self ):
        return "INDUS WHAM"

    ## INSTANCE PREPARING WHAM
    def prep_wham( self ):
        """FUNCTION PREPARING FOR WHAM"""
        ## PATH UMBRELLA
        self.umbrella_dir = os.path.join( self.sim_working_dir, "umbrella" )
        
        ## PATH OUTPUT
        self.output_dir = os.path.join( self.sim_working_dir, "output_files" )
        
        ## PATH WHAM
        self.wham_dir = os.path.join( self.sim_working_dir, "wham" )
        
        ## WHAM FILE
        wham_file = os.path.join( self.wham_dir, self.input_prefix + "_wham.csv" )
        
        ## SET WHAM INPUT
        self.wham_input[-1] = wham_file
        
        ## CREATE WHAM PATH (IF NOT EXISTS)
        if os.path.exists( self.wham_dir ) is not True:
            os.mkdir( self.wham_dir )

        ## ADD METADATA FILE
        self.metadata_file = os.path.join( self.wham_dir, self.input_prefix + "_metadata.dat" )
        
        ## PATH WHAM LOG
        self.wham_log = os.path.join( self.wham_dir, "wham.log" )
        
        ## GATHER INPUT FILES
        num_waters      = []
        in_files        = []
        truncated_files = []
        histogram_files = []
        
        ## LOOP THROUGH UMBRELLA DIRECTORIES
        for sub_dir in os.listdir( self.umbrella_dir ):
            ## ENTER IF INDUS_ PREFIX
            if "indus_" in sub_dir:
                ## GET WATERS
                water = sub_dir.split( "indus_" )[-1]
                num_waters.append( float( water ) )
                
                ## LOOP THROUGH SUB DIRECTORY
                subdir = os.path.join( self.umbrella_dir, sub_dir )
                for filename in os.listdir( subdir ):
                    ## FIND INDUS INPUTS
                    if "water_num" in filename \
                       and "truncated_" not in filename:
                        ## ADD FILE TO LIST
                        in_file = os.path.join( subdir, filename )
                        in_files.append( in_file )
                        truncated_file = os.path.join( subdir, "truncated_" + filename )
                        truncated_files.append( truncated_file )
                    
                    ## FIND HISTOGRAM FILES
                    if "histo_" in filename:
                        ## ADD FILE TO LIST
                        hist_file = os.path.join( subdir, filename )
                        histogram_files.append( hist_file )                        
        
        ## CREATE SORTING KEY
        key = lambda func: float( func.rsplit( os.path.extsep, 1 )[0].rsplit( ".", 1 )[-1] )
        
        ## ADD TO CLASS
        self.in_files = sorted( in_files, key = key )
        
        ## ADD TRUNCATED FILES CLASS
        self.truncated_files = sorted( truncated_files, key = key )

        ## CREATE SORTING KEY
        key = lambda func: float( func.rsplit( os.path.extsep, 1 )[0].rsplit( "_", 1 )[-1] )
        
        ## ADD HISTOGRAM FILES TO CLASS
        self.histogram_files = sorted( histogram_files, key = key )
        
        ## ADD NUM WATERS
        self.num_waters = sorted(num_waters)

        ## SET WHAM BOUNDS (ASSUME STEP OF 1)
        self.range  = ( self.num_waters[0]-0.5, self.num_waters[-1]+0.5 )
        self.n_bins = np.arange( self.range[0], self.range[-1], 1 ).size
        
    ## INSTANCE CREATING WHAM HISTOGRAMS
    def histograms( self ):
        """FUNCTION TO CREATE WHAM HISTOGRAMS""" 
        ## PATH TO PKL
        out_name = self.input_prefix + "_wham_histograms.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )
        
        if self.recompute_wham is True or \
           os.path.exists( path_pkl ) is not True:       
            ## LOAD DATA FROM INPUT FILES
            for ii, hist_file in enumerate( self.histogram_files ):
                ## LOAD CSV
                data = load_csv( hist_file )
                
                ## CREATE DATA OBJECT IF FIRST ITERATION
                if ii < 1:
                    ## STORE DATA, LAST COLUMN NOT NEEDED
                    histogram_data = data[:,:2]
                    
                else:
                    histogram_data = np.hstack(( histogram_data, data[:,1][:,np.newaxis] ))
            
            ## SAVE PKL
            save_pkl( histogram_data, path_pkl )
        else:
            ## LOAD PKL
            histogram_data = load_pkl( path_pkl )
            
        ## RESULTS
        self.hist_N     = histogram_data[:,0]
        self.histograms = histogram_data[:,1:]
    
    ## INSTANCE TRUNCATING INDUS OUTPUT
    def truncate( self, time_range = ( 0, -1 ) ):
        """FUNCTION TO TRUNCATE INDUS OUTPUT"""
        ## EXTRACT FRAME RANGE
        start = time_range[0]
        end   = time_range[-1]
        
        ## LOAD DATA FROM INPUT FILES
        for in_file, truncated_file in zip( self.in_files, self.truncated_files ):
            ## LOAD CSV
            data = load_csv( in_file )

            ## DETERMINE ENDING INDEX
            if end > 0:
                end_ndx = np.argmin( np.abs(data[:,0] - end) )
            else:
                end_ndx = end
    
            ## DETERMINE STARTING INDEX
            start_ndx = np.argmin( np.abs(data[:,0] - start) )
            
            ## TRUNCATE DATA
            truncated_data = data[start_ndx:end_ndx,:]
            
            ## WRITE TRUNCATED DATA TO FILE
            write_csv( truncated_data, truncated_file )
    
    ## INSTANCE CREATING METADATA FILE (INPUT TO GROSSFIELD WHAM)
    def metadata( self ):
        """FUNCTION TO CREATE META DATA FILE"""
        ## OPEN FILE AND MAKE WRITABLE
        outfile = open( self.metadata_file, 'w+' )
        
        ## WRITE FILE LINE BY LINE
        for trunc_file, num in zip( self.truncated_files, self.num_waters ):
            if num < 10:
                outfile.write( "{} {} {} {} {} \n".format( trunc_file, num, num, 0.0, self.spring_strong ) )
            else:
                outfile.write( "{} {} {} {} {} \n".format( trunc_file, num, num, 0.0, self.spring_weak ) )
                
        ## CLOSE FILE
        outfile.close()
        
        ## PRINTING    
        print( "  METADATA FILE SAVED TO {}\n".format( self.metadata_file ) )

    ## INSTANCE TO COMPUTE WHAM
    def wham( self, wham_file ):
        """FUNCTION TO COMPUTE GROSSFIELD WHAM"""
        ## CREATE BASH CMD STRING        
        cmd = "{} Px=0 {} {} {} Py=0 {} {} {} {} {} {} {} {} {} > {}".format(
              WHAM,
              self.range[0],
              self.range[-1],
              self.n_bins,
              self.range[0],
              self.range[-1],
              self.n_bins,
              self.tol,
              self.temperature,
              0,
              self.metadata_file,
              wham_file,
              0,
              self.wham_log )
                
        ## RUNNING SUBPROCESS
        print( "RUNNING WHAM...\n" )
        p = subprocess.Popen( cmd,
                              stdin  = subprocess.PIPE,
                              stdout = subprocess.PIPE,
                              stderr = subprocess.PIPE,
                              cwd    = self.wham_dir,
                              shell  = True )
        
        ## RUN WITH NO INPUTS
        (output, err) = p.communicate()
        
        ## WAITING FOR COMMAND TO FINISH
        p.wait()
        
        ## PRINT OUTPUT PATH
        print( "WHAM WRITTEN TO {}".format( wham_file ) )

    ## INSTANCE ANALYZING WHAM OUTPUT
    def compute( self, wham_input = [ 0, -1, None ] ):
        """FUNCTION TO PUT WHAM OUTPUT INTO READABLE FORMAT"""   
        ## CHECK IF CLASS INPUT
        if None in wham_input:
            ## EXTRACT INPUTS
            time_range = self.wham_input[:2]
            wham_file  = self.wham_input[-1]
        else:
            ## EXTRACT INPUTS
            time_range = wham_input[:2]
            wham_file  = wham_input[-1]
                    
        if self.recompute_wham is True or \
           os.path.exists( wham_file ) is not True:            
            ## TRUNCATE FILES
            self.truncate( time_range )
                        
            ## CREATE METADATA FILE
            self.metadata()
            
            ## RUN WHAM
            self.wham( wham_file )
    
## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE CLASS
    indus = IndusWham( **kwargs )

    ## COMPUTE WHAM HISTOGRAMS
    indus.histograms()
    
    ## RESULTS
    hist_n     = indus.hist_N
    histograms = indus.histograms

    ## COMPUTE WHAM
    indus.compute()
    
    ## RETURN RESULTS
    return { "wham_hist_N"     : hist_n,
             "wham_histograms" : histograms, }
        
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
    
    ## INDUS WHAM
    indus = IndusWham( sim_working_dir    = path_traj,
                       input_prefix       = input_prefix,
                       start              = 2000.,
                       end                = -1,
                       spring_weak        = 2.0,
                       spring_strong      = 8.5,
                       tol                = 0.00001,
                       temperature        = 300,
                       recompute_wham     = True,
                       verbose            = True, )

    ## COMPUTE WHAM HISTOGRAMS
    indus.histograms()
    
    ## RESULTS
    hist_N     = indus.hist_N
    histograms = indus.histograms
            
    ## COMPUTE WHAM
    indus.compute()
    