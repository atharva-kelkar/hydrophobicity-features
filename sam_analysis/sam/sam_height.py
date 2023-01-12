"""
sam_height.py 
script contains various methods to determines the height of a SAM

CREATED ON: 09/25/2020

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

## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORT MISC TOOLS
from sam_analysis.core.misc_tools import compute_com, end_group_atom_indices, fixed_grid
## WILLARD-CHANDLER CLASS
from sam_analysis.water.willard_chandler import WillardChandler

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## CLASS COMPUTING HEIGHT
class SamHeight:
    """class object used to compute the height of a SAM"""
    def __init__( self,
                  traj             = None,
                  sim_working_dir  = None,
                  input_prefix     = None,
                  recompute_height = False,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.traj             = traj
        self.sim_working_dir  = sim_working_dir
        self.input_prefix     = input_prefix
        self.recompute_height = recompute_height
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)
                
    ## METHOD COMPUTING HEIGHT AS AVG END GROUP COM HEIGHT OF LIGANDS
    def average_end_group( self, per_frame = False ):
        ## PATH TO PKL
        out_name = self.input_prefix + "_end_group_heights.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )
        
        ## LOAD DATA
        if self.recompute_height is True or \
           os.path.exists( path_pkl ) is not True:
            ## LOAD TRAJ IN NOT INPUT
            if self.traj is None:
                self.traj = load_md_traj( path_traj    = self.sim_working_dir,
                                          input_prefix = self.input_prefix )
                
            ## GET END GROUP ATOM INDICES
            end_atom_indices = end_group_atom_indices( self.traj, labels = False )
            
            ## COMPUTE COM POSITION OF END GROUP
            end_group_coms = np.zeros( shape = ( self.traj.n_frames, 
                                                 len(end_atom_indices), 
                                                 3 ) )
            for ii, indices in enumerate(end_atom_indices):
                end_group_coms[:,ii,:] = compute_com( self.traj, indices )
                
            ## GET HEIGHTS
            heights = end_group_coms[...,2]
            
            ## SAVE HEIGHT DATA
            save_pkl( heights, path_pkl )
        else:
            ## LOAD HEIGHTS
            heights = load_pkl( path_pkl )
                
        ## OUTPUT SURFACE AVERAGE PER FRAME
        if per_frame is True:
            return heights.mean( axis = 1 )
        
        ## OUTPUT SURFACE ENSEMBLE AVERAGE
        return heights.mean()
    
    ## METHOD COMPUTING HEIGHT AS AVG WILLARD-CHANDLER INTERFACE
    def willard_chandler( self, per_frame = False ):
        ## COMPUTE WILLARD-CHANDLER GRID
        willard_chandler = WillardChandler( **self.__dict__ )
        ## COMPUTE GRID
        grid = willard_chandler.grid( per_frame = per_frame )
        
        if per_frame is True:
            ## HEIGHT PLACE HOLDER
            height = []
            ## LOOP THROUGH GRIDS
            for gg in grid:                
                ## CALCULATE HEIGHTS (height = avg(grid))
                height.append( np.nanmean(gg[:,2]) )
            
            ## CONVERT TO NUMPY ARRAY
            return np.array(height)
            
        ## CALCULATE HEIGHTS (height = avg(grid))
        height = np.nanmean( grid[:,2] )
        
        ## RETURN RESULTS
        return height
    
    ## METHOD COMPUTING HEIGHT AS AVG WILLARD-CHANDLER INTERFACE
    def instantaneous_height( self, ):
        ## COMPUTE WILLARD-CHANDLER GRID
        willard_chandler = WillardChandler( **self.__dict__ )
        ## COMPUTE GRID
        wc_grid = willard_chandler.grid( per_frame = False )
        
        ## LOAD TRAJ IN NOT INPUT
        if self.traj is None:
            self.traj = load_md_traj( path_traj    = self.sim_working_dir,
                                      input_prefix = self.input_prefix )

        ## GET END GROUP ATOM INDICES
        end_atom_indices = end_group_atom_indices( self.traj, labels = False )
        
        ## COMPUTE COM POSITION OF END GROUP
        end_group_coms = np.zeros( shape = ( self.traj.n_frames, 
                                             len(end_atom_indices), 
                                             3 ) )
        for ii, indices in enumerate(end_atom_indices):
            end_group_coms[:,ii,:] = compute_com( self.traj, indices )

        ## COMPUTE TIME AVERAGE COM
        end_group_grid = fixed_grid( end_group_coms.mean( axis = 0 ), 
                                     xrange   = [ 0, self.traj.unitcell_lengths[0,0] ],
                                     yrange   = [ 0, self.traj.unitcell_lengths[0,1] ],
                                     spacing  = 0.1,
                                     periodic = True )
        
        ## COMPUTE HEIGHT DIFFERENCE
        grid = np.zeros_like( wc_grid )
        grid[:,:2] = wc_grid[:,:2]
        grid[:,2] = wc_grid[:,2] - end_group_grid[:,2]
        
        ## RETURN RESULTS
        return grid

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE CLASS
    height_obj = SamHeight( **kwargs )

    ## AVERAGE END GROUP HEIGHT
    height_end_group = height_obj.average_end_group( per_frame = False )
    ## PER FRAME
    height_end_group_per_frame = height_obj.average_end_group( per_frame = True )
    ## WILLARD-CHANDLER INTERFACE
    height_wc = height_obj.willard_chandler( per_frame = False )
    ## PER FRAME
    height_wc_per_frame = height_obj.willard_chandler( per_frame = True )
    ## INSTANTANEOUS
    height_instant = height_obj.instantaneous_height()
    
    ## RETURN RESULTS
    return { "height_end_group"                  : height_end_group,
             "height_end_group_per_frame"        : height_end_group_per_frame,
             "height_willard_chandler"           : height_wc,
             "height_willard_chandler_per_frame" : height_wc_per_frame,
             "height_instantaneous"              : height_instant, }
    
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
    sam_dir = r"ordered_ch3_sam"
    
    ## WORKING DIR
    working_dir = os.path.join(test_dir, sam_dir)
    
    ## LOAD TRAJECTORY
    path_traj = check_server_path( working_dir )

    ## TRAJ PREFIX
    input_prefix = "sam_prod"
    
    ## LOAD TRAJECTORY
    traj = load_md_traj( path_traj    = path_traj,
                         input_prefix = input_prefix )

    ## WILLARD CHANDLER ARGUMENTS
    wc_args = { "n_procs"              : 4,
                "alpha"                : 0.24,
                "contour"              : 16.0,
                "mesh"                 : [0.1, 0.1, 0.1],
                "recompute_interface"  : False,
                "print_freq"           : 10 }

    ## INITIALIZE CLASS
    height_obj = SamHeight( traj             = traj,
                            sim_working_dir  = path_traj,
                            input_prefix     = input_prefix,
                            recompute_height = True,
                            **wc_args )

    ## AVERAGE END GROUP HEIGHT
    height_end_group = height_obj.average_end_group( per_frame = False )
    
    ## PER FRAME
    height_end_group_per_frame = height_obj.average_end_group( per_frame = True )
    
    ## WILLARD-CHANDLER INTERFACE
    height_wc = height_obj.willard_chandler( per_frame = False )
    
    ## PER FRAME
    height_wc_per_frame = height_obj.willard_chandler( per_frame = True )
    
    ## PRINT OUT RESULTS
    print( "\nAVG END GROUP HEIGHT {}".format( height_end_group ) )
    print( "END GROUP PER FRAME SIZE {}".format( height_end_group_per_frame.shape ) )
    print( "WC INTERFACE HEIGHT {}".format( height_wc ) )
    print( "WC INTERFACE PER FRAME SIZE {}".format( height_wc_per_frame.shape ) )
    
           