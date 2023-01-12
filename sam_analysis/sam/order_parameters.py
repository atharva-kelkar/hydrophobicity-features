"""
order_parameters.py 
script contains various order parameter calculations

CREATED ON: 10/22/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:
"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT PYTHON MODULES
import numpy as np
import mdtraj as md

##############################################################################
# METHODS
##############################################################################
### FUNCTION TO COMPUTE DIHEDRAL ANGLES
def compute_dihedral_angles( traj, dihedral_list, periodic = True ):
    '''
    The purpose of this function is to calculate all the dihedral angles given a list
    INPUTS:
        self: class object
        traj: trajectory from md.traj
        dihedral_list:[list] list of list of all the possible dihedrals
        periodic: [logical] True if you want periodic boundaries
    OUTPUTS:
       dihedrals: [np.array, shape=(time_frame, dihedral_index)]  dihedral angles in degrees from 0 to 360 degrees
    '''
    ## CALCULATING DIHEDRALS FROM MDTRAJ
    dihedrals = md.compute_dihedrals(traj, dihedral_list, periodic = periodic )
    # RETURNS NUMPY ARRAY AS A SHAPE OF: (TIME STEP, DIHEDRAL) IN RADIANS
    dihedrals = np.rad2deg(dihedrals)
    dihedrals[ dihedrals < 0 ] = dihedrals[ dihedrals < 0 ] + 360  # to ensure every dihedral is between 0-360
    
    ## GET DIHEDRAL FRACTIONS
    fractions = np.logical_and( dihedrals < 240, dihedrals > 120 )
    
    ## COMPUTE AVERAGE PER FRAME
    dihedrals_per_frame = np.mean( fractions, axis = 1 )
    
    return dihedrals_per_frame

## FUNCTION TO CALCULATE PSI
def compute_psi( traj, point, all_points, cutoff, periodic ):
    R'''
    Calculates hexatic order of a given point    
    '''
    psi = np.nan * np.ones( shape = traj.time.size, dtype = 'complex' )
    ## DEFINE REFERENCE
    ref_vector = np.array([ 1, 0 ])
    ref_mag = np.sqrt( ref_vector[0]**2 + ref_vector[1]**2 )
    
    ## MAKE LIST OF POTENTIAL NEIGHBOR ATOMS
    potential_neighbors = np.array([ [ ii, point ] for ii in all_points if ii != point ])
    vector = md.compute_displacements( traj, potential_neighbors, periodic = periodic )
    dist = np.sqrt( np.sum( vector**2., axis = 2 ) )
    
    ## DETERMINE ATOMS IN CUTOFF
    dist[abs( dist ) > cutoff] = 0.
    mask = abs( dist ) > 0.
    n_neighbors = mask.sum( axis = 1 )
    
    ## CALCULATE ANGLE USING DOT PRODUCT AND DETERMINANT
    theta = np.zeros( shape = dist.shape )
    exp_theta = np.zeros( shape = dist.shape, dtype = 'complex' )
    dot_vec = vector[:,:,0] * ref_vector[0] + vector[:,:,1] * ref_vector[1]
    det_vec = vector[:,:,0] * ref_vector[1] - vector[:,:,1] * ref_vector[0]
    theta[mask] = np.arccos( dot_vec[mask] / ( dist[mask] * ref_mag ) )
    theta[det_vec < 0.] = 2. * np.pi - theta[det_vec < 0.]
    exp_theta[mask] = np.exp( 6j * theta[mask] )
    psi[n_neighbors > 0.] = np.sum( exp_theta[n_neighbors > 0.], axis = 1 ) / n_neighbors[n_neighbors > 0.]
    
    return np.abs( psi )**2.