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
from sam_analysis.core.misc_tools import compute_com, end_group_atom_indices

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## CLASS COMPUTING HEIGHT
class EndGroupDensity:
    """class object used to compute the height of a SAM"""
    def __init__( self,
                  sim_working_dir       = None,
                  input_prefix          = None,
                  recompute_sam_density = False,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.sim_working_dir       = sim_working_dir
        self.input_prefix          = input_prefix
        self.recompute_sam_density = recompute_sam_density
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)
                
    ## METHOD COMPUTING DENSITY AS END GROUP COM HISTOGRAM
    def density( self, per_frame = False ):
        ## PATH TO PKL
        out_name = self.input_prefix + "_end_group_coms.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )
        
        ## LOAD DATA
        if self.recompute_sam_density is True or \
           os.path.exists( path_pkl ) is not True:
            ## LOAD TRAJ
            traj = load_md_traj( path_traj    = self.sim_working_dir,
                                 input_prefix = self.input_prefix )
            
            ## GET UNITCELL
            xrange = ( 0, traj.unitcell_lengths[0,0] )
            yrange = ( 0, traj.unitcell_lengths[0,1] )
            zrange = ( 0, traj.unitcell_lengths[0,2] )
            xyz_range = [ xrange, yrange, zrange ]

            ## GET END GROUP ATOM INDICES
            eg_indices, eg_labels = end_group_atom_indices( traj, labels = True )

            ## GET ONLY POLAR GROUPS
            end_atom_indices = []
            for ii in range(len(eg_indices)):
                if eg_labels[ii] != "DOD":
                    end_atom_indices.append( eg_indices[ii] )
            
            ## COMPUTE COM POSITION OF END GROUP
            end_group_coms = np.zeros( shape = ( traj.n_frames, 
                                                 len(end_atom_indices), 
                                                 3 ) )
            for ii, indices in enumerate(end_atom_indices):
                end_group_coms[:,ii,:] = compute_com( traj, indices )
            
            ## SAVE HEIGHT DATA
            save_pkl( [ end_group_coms, xyz_range ], path_pkl )
        else:
            ## LOAD HEIGHTS
            data = load_pkl( path_pkl )
            
            ## UNPACK DATA
            end_group_coms = data[0]
            xyz_range      = data[1]

        self.end_group_coms = end_group_coms
        self.xyz_range = xyz_range

        ## X
        bin_width = 0.4
        x_range = ( xyz_range[0][0], xyz_range[0][1] )
        self.x = np.arange( x_range[0], x_range[1], bin_width )

        ## COMPUTE TOTAL DISTRIBUTION
        y = np.histogram( end_group_coms[:,:,0], bins = len(self.x) )[0]

        ## NORMALIZE
        # ALL
        constant  = np.max([ np.trapz( y, dx = bin_width ), 1 ])
        self.dist = y / constant  

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE CLASS
    obj = EndGroupDensity( **kwargs )

    ## COMPUTE DIST
    obj.density()
    
    ## RETURN RESULTS
    return { "end_group_density_x"    : obj.x,
             "end_group_density_dist" : obj.dist, }
    
#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## IMPORT CHECK SERVER PATH
    from sam_analysis.core.check_tools import check_server_path

    ## TESTING DIRECTORY
    test_dir = r"/home/bdallin/simulations/polar_sams/unbiased"
    
    ## SAM DIRECTORY
    sam_dir    = r"sam_single_12x12_separated_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36"
    sample_dir = r"sample1"

    ## TRAJ NAME
    traj_name   = r"sam_prod"
    
    ## WORKING DIR
    working_dir = os.path.join(test_dir, sam_dir, sample_dir)
    working_dir = check_server_path( working_dir )

    ## INITIALIZE WATER POSITION
    obj = EndGroupDensity( sim_working_dir       = working_dir,
                           input_prefix          = traj_name,
                           recompute_sam_density = True, )

    ## COMPUTE COORDINATION DIST
    obj.density()

    x  = obj.x
    y  = obj.dist
    
    # ## PRINT OUT RESULTS
    # print( "COORDINATION ARRAY SIZE: {}".format( coord.shape ) )    
           