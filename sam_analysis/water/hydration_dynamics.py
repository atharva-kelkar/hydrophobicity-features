"""
hydration_dynamics.py 
script contains functions to compute water dynamics in a MD trajectory

CREATED ON: 12/02/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)

** NOTES **
PLEASE DOUBLE CHECK ANY VALUES COMPUTED FROM THIS TOOL. I HAVEN'T BEEN ABLE 
TO VALIDATE YET. WILL LIKELY NEED TO USE A LARGER TRAJECTORY.
    
** UPDATES **

TODO:
"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT OS
import os
## IMPORT COPY
import copy
## IMPORT NUMPY
import numpy as np

## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj
## IMPORT MISC TOOLS
from sam_analysis.core.misc_tools import compute_displacements, compute_com, \
                                         fit_exp

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## CLASS TO COMPUTE RESIDENCE TIME
class ResidenceTime:
    """class object used to compute density profiles"""
    def __init__( self,
                  traj            = None,
                  sim_working_dir = None,
                  input_prefix    = None,
                  center          = [ 0., 0., 0. ],
                  dimensions      = [ 2., 2., 0.3 ],
                  water_residues  = [ "SOL", "HOH" ],
                  n_procs         = 1,
                  periodic        = True,
                  recompute       = False,
                  verbose         = True,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.traj            = traj
        self.sim_working_dir = sim_working_dir
        self.input_prefix    = input_prefix
        self.center          = center
        self.dimensions      = dimensions
        self.water_residues  = water_residues
        self.n_procs         = n_procs
        self.periodic        = periodic
        self.recompute       = recompute
        self.verbose         = verbose
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)
        
    ## COMPUTE RESIDENCE TIME
    def tau( self, cutoff = 50 ):
        """Function computes tau by fitting ACF"""
        ## COMPUTE ACF
        profile = self.profile()
        
        ## ONLY LOOK AT FIRST 50 FRAMES FOR FIT
        tau = fit_exp( self.traj.time[1:cutoff], profile[1:cutoff] )
        
        ## RETURN RESULTS
        return tau["tau"]
    
    ### COMPUTE RESIDENCE TIME PROFILE (ACF)
    def profile( self, com = True ):
        """Function computes the hydration residence time autocorrelation function"""
        ## CREATE COPY OF TRAJ
        traj = copy.deepcopy(self.traj)
        
        ## GET RESIDUES
        if com is True:
            ## GET ATOM GROUPS TO COMPUTE DENSITY
            residue_group_indices = [ [ atom.index for atom in residue.atoms ] 
                                         for residue in traj.topology.residues 
                                         if residue.name in self.water_residues ]
            atom_indices = np.array([ residue[0] for residue in residue_group_indices ])
                    
            ## UPDATE TRAJ SO HEAVY ATOM HAS COM POSITION
            for ndx, res_indices in zip( atom_indices, residue_group_indices ):
                traj.xyz[:,ndx,:] = compute_com( traj, res_indices )
        else:
            ## EXTRACT HEAVY ATOM INDICES IN RESIDUE LIST
            atom_indices = np.array([ atom.index for atom in traj.topology.atoms 
                                       if atom.residue.name in self.water_residues 
                                       and atom.element.symbol == "O" ])
        
        ## GATHER POSITIONS OF WATER ATOMS RELATIVE TO SHAPE CENTER
        dist = compute_displacements( traj,
                                      atom_indices   = atom_indices,
                                      box_dimensions = self.traj.unitcell_lengths,
                                      ref_coords     = self.center,
                                      periodic       = self.periodic )
        
        ## DEFINE UPPER AND LOWER BOUNDS
        x_range = np.array([ -0.5, 0.5 ]) * self.dimensions[0]
        y_range = np.array([ -0.5, 0.5 ]) * self.dimensions[1]
        z_range = np.array([ -1, 1 ]) * self.dimensions[2]
        
        ## APPLY HEAVISIDE STEP FUNCTION (1 IF IN CAVITY, 0 OTHERWISE)
        x_heaviside = np.logical_and( dist[...,0] > x_range[0], dist[...,0] < x_range[-1] )
        y_heaviside = np.logical_and( dist[...,1] > y_range[0], dist[...,1] < y_range[-1] )
        z_heaviside = np.logical_and( dist[...,2] > z_range[0], dist[...,2] < z_range[-1] )
        
        ## COMBINE HEAVISIDE FUNCTIONS
        theta_heaviside = x_heaviside * y_heaviside * z_heaviside
                
        ## COMPUTE AUTOCORRELATION FUNCTION
        C_res = acf( theta_heaviside )
        
        ## RETURN RESULTS    
        return C_res

## FUNCTION COMPUTES AUTOCORRELATION FUNCTION
def acf( X ):
    '''Function computes an autocorrelation function of the data'''
    ## GET NUM ROWS
    n = X.shape[0]
    
    ## CREATE PLACE HOLDER
    C = np.zeros(n)
    
    ## LOOP THROUGH COLUMNS
    nn = 0.
    for n_col in range(X.shape[1]):
        ## COMPUTE ACF FOR EACH COLUMN
        x = X[:,n_col]
        
        ## SKIP IF ALL EQUAL
        if x.std() > 0.:
            ## NORMALIZE INPUTS
            a = ( x - x.mean() ) / x.std() / x.size
            b = ( x - x.mean() ) / x.std()
            
            ## DETERMINE AUTOCORRELATION
            C += np.correlate( a, b, mode = 'full' )[-n:]
            
            ## UPDATE COUNTER
            nn += 1.
            
    ## RETURN NORMALIZED RESULTS
    return C / nn

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## IMPORT WILLARD-CHANDLER HEIGHT FUNCTION
    from sam_analysis.water.willard_chandler import WillardChandler
    ## IMPORT CHECK SERVER PATH
    from sam_analysis.core.check_tools import check_server_path
    ## TESTING DIRECTORY
    test_dir = r"/mnt/r/python_projects/sam_analysis/sam_analysis/testing"
    
    ## SAM DIRECTORY
    sam_dir = r"ordered_nonpolar_sam"
    
    ## WORKING DIR
    working_dir = os.path.join(test_dir, sam_dir)
    
    ## LOAD TRAJECTORY
    path_traj = check_server_path( working_dir )

    ## LOAD TRAJECTORY
    input_prefix = "sam_prod"
    
    ## LOAD TRAJ
    traj = load_md_traj( path_traj    = path_traj,
                         input_prefix = input_prefix )
    
    ## CALCULATE WILLARD-CHANDLER GRID
    willard_chandler = WillardChandler( traj            = traj,
                                        sim_working_dir = path_traj,
                                        input_prefix    = input_prefix,
                                        n_procs         = 4,
                                        alpha           = 0.24,
                                        contour         = 16.0,
                                        mesh            = [0.1, 0.1, 0.1],
                                        recompute       = False,
                                        print_freq      = 10, )
    ## STORE GRID
    grid = willard_chandler.grid()
    
    # AVERAGE GRID TO GET HEIGHT REFERENCE
    wc_mean = grid[:,2].mean()
    
    ## WATER ORIENTATION
    dynamics_obj = ResidenceTime( traj            = traj,
                                  sim_working_dir = path_traj,
                                  input_prefix    = input_prefix,
                                  z_ref           = wc_mean,
                                  center          = [ 0.5*traj.unitcell_lengths[0,0],
                                                      0.5*traj.unitcell_lengths[0,1],
                                                      wc_mean+2.5 ],
                                  dimensions      = [ traj.unitcell_lengths[0,0],
                                                      traj.unitcell_lengths[0,1],
                                                      0.5 ],
                                  water_residues  = [ "SOL", "HOH" ],
                                  periodic        = True,
                                  recompute       = True,
                                  verbose         = True )    
    ## COMPUTE PROFILE
    profile = dynamics_obj.profile( com = True )
    ## PRINT OUT SIZE
    print( "PROFILE COMPLETE: {}\n".format( str(profile.sum() > 0) ) )
    
    ## COMPUTE RESIDENCE TIME
    residence_time = dynamics_obj.tau( cutoff = 20 )
    ## PRINT OUT SIZE
    print( "RESIDENCE TIME: {}\n".format( residence_time ) )
