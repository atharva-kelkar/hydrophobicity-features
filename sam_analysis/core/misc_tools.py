"""
misc_tools.py 
script contains various methods to reduce redundancies

CREATED ON: 09/26/2020

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
import pandas as pd
from datetime import datetime

## IMPORT SCIPY.INTERPOLATE
from scipy.interpolate import griddata

## IMPORT GLOBAL LIGAND AND SOLVENT DATA
from sam_analysis.globals.ligands import LIGAND_END_GROUPS
from sam_analysis.globals.solvents import SOLVENTS

##############################################################################
## FUNCTIONS AND CLASSES
##############################################################################
### CLASS FUNCTION TO TRACK TIME
class track_time:
    '''
    The purpose of this function is to track time.
    INPUTS:
        void
        
    FUNCTIONS:
        time_elasped: 
            function to print total time elapsed
    '''
    def __init__(self):
        ## START TRACKING TIME
        self.start = datetime.now()
        return
    ## FUNCTION TO PRINT TIME
    def time_elasped(self, prefix_string = None):
        ''' Function to print time elapsed '''
        time_elapsed = datetime.now() - self.start
        if prefix_string is None:
            print( 'Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) )
        else:
           print( '{} time elapsed'.format( prefix_string ) + ' (hh:mm:ss.ms) {}'.format( time_elapsed ) ) 
        return
    
    ## UPGRADING NAME
    def time_elapsed(self, **args):
        self.time_elasped(**args)
        return

### FUNCTION THAT SPLITS A LIST INTO MULTIPLE PARTS
def split_list(alist, wanted_parts=1):
    '''
    The purpose of this function is to split a larger list into multiple parts
    INPUTS:
        alist: [list] original list
        wanted_parts: [int] number of splits you want
    OUTPUTS:
        List containing chunks of your list
    Reference: https://stackoverflow.com/questions/752308/split-list-into-smaller-lists?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    '''
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

## CREATE ITERABLE
def combine_objects( obj1, obj2 ):
    """create an iterable object from two objects"""
    for one, two in zip( obj1, obj2 ):
        yield ( one, two )

## FUNCTION DETERMINING LIGAND ATOM INDICES
def ligand_heavy_atom_indices( traj, labels = False ):
    ## GET END GROUP ATOM INDICES
    ligand_atom_indices = []
    ligand_labels       = []
    ## LOOP THROUGH LIGANDS
    for ligand in traj.topology.residues:
        ## LOOP THROUGH LIGAND ATOMS IF LIGAND HAS END GROUP
        if ligand.name not in SOLVENTS.keys():
            atom_indices = []
            for atom in ligand.atoms:
                if atom.element.symbol != "H":
                    ## APPEND ATOMS
                    atom_indices.append( atom.index )
            ## APPEND INDICES TO LIGAND
            ligand_atom_indices.append( atom_indices )
            ## APPEND NAME TO LIGAND
            ligand_labels.append( ligand.name )
    
    ## RETURN LABELS IF TRUE
    if labels is True:
        return ligand_atom_indices, ligand_labels
    
    ## RETURN LIGAND INDICES ONLY IF LABELS FALSE
    return ligand_atom_indices   

## FUNCTION DETERMINING LIGAND END GROUP INDICES       
def end_group_atom_indices( traj, labels = False ):
    ## GET END GROUP ATOM INDICES
    ligand_atom_indices = []
    ligand_labels       = []
    ## LOOP THROUGH END GROUPS
    for eg_name, eg_atoms in LIGAND_END_GROUPS.items():
        ## LOOP THROUGH LIGANDS
        for ligand in traj.topology.residues:
            ## LOOP THROUGH LIGAND ATOMS IF LIGAND HAS END GROUP
            if ligand.name == eg_name:
                atom_indices = []
                for atom in ligand.atoms:
                    ## APPEND ATOMS IF ATOM MATCH END GROUP
                    if atom.name in eg_atoms:
                        atom_indices.append( atom.index )
                ## APPEND INDICES TO LIGAND
                ligand_atom_indices.append( atom_indices )
                ## APPEND NAME TO LIGAND
                ligand_labels.append( eg_name )
    
    ## RETURN LABELS IF TRUE
    if labels is True:
        return ligand_atom_indices, ligand_labels
    
    ## RETURN LIGAND INDICES ONLY IF LABELS FALSE
    return ligand_atom_indices    

## FUNCTION COMPUTING COM FOR EACH FRAME OF A TRAJECTORY
def compute_com( traj, atom_indices ):
    ## Compute the center of mass of the atom group provided in atom_indices
    group_masses = np.array([ traj.topology.atom(ii).element.mass for ii in atom_indices ])
    group_mass = group_masses.sum()
    coords = traj.xyz[ :, atom_indices, : ]
    com = np.sum( coords * group_masses[np.newaxis,:,np.newaxis], axis=1 ) / group_mass
    
    return com

### FUNCTION TO FIND DIHEDRAL LIST
def dihedral_lists( indices ):
    R'''
    The purpose of this function is to find all dihedral indices. 
    '''
    ## CREATING A BLANK LIST
    dihedral_list = []
    ## GENERATING DIHEDRAL LIST BASED ON HEAVY ATOMS
    for ligand in indices:
        ## ONLY CHECKS LIGINDS WITH THE TAIL GROUP INSIDE THE CONTACT AREA
        ## LOOPING THROUGH TO GET A DIHEDRAL LIST (Subtract owing to the fact you know dihedrals are 4 atoms)
        for each_iteration in range(len(ligand)-3):
            ## APPENDING DIHEDRAL LIST
            dihedral_list.append(ligand[each_iteration:each_iteration+4])
                
    return dihedral_list

### FUNCTION TO COMPUTE DISPLACEMENTS FROM POINT FOR NVT ENSEMBLE
def compute_displacements( traj, 
                           atom_indices,
                           box_dimensions,
                           ref_coords = [0,0,0], 
                           periodic = True ):
    R'''
    Function to compute distances between points in a simulation trajectory. Uses
    numpy vectorization to speed up calculations. Can handle single or multi-frame
    trajectories and box dimensions.
    NOTE: traj must be NVT
    
    INPUT:
        traj: [mdtraj.traj]
            trajectory loaded from mdtraj
        atom_indices: [list]
            list of atom indices to compute distances between atom and reference point
        ref_coords: [list]
            list containing x,y,z reference coordinates
        periodic: [bool]
            True compute with pbc, False compute w/o pbc
            
    OUTPUT:
        distances: [numpy.array]
            NxM array where N is number of frames and M is number of atoms
    '''    
    def pbc_distances( x0, x1, dimensions, periodic ):
        ## compute distances, assumes static reference point in box
        delta = x0 - x1
        if periodic is True:
            # if 1D: single frame
            if len(dimensions.shape) == 1:
                ## ADJUST FOR PERIODIC BOUNDARY CONDITIONS
                delta = np.where( delta > 0.5 * dimensions, delta - dimensions, delta )
                delta = np.where( delta < -0.5 * dimensions, delta + dimensions, delta )
            # if 2D: multiple frames
            else:
                ## ADJUST FOR PERIODIC BOUNDARY CONDITIONS
                delta = np.where( delta > 0.5 * dimensions[:,np.newaxis,:], delta - dimensions[:,np.newaxis,:], delta )
                delta = np.where( delta < -0.5 * dimensions[:,np.newaxis,:], delta + dimensions[:,np.newaxis,:], delta )
                
        return delta
    
    xyz = traj.xyz[:,atom_indices,:]
    return pbc_distances( xyz, np.array(ref_coords), box_dimensions, periodic ).squeeze()

## METHOD TO INTERPOLATE A SURFACE ON A FIXED XY GRID
def fixed_grid( irr_grid, xrange = [], yrange = [], spacing = 0.1, periodic = False ):        
    ## EXTRACT DATA FOR CLARITY
    x = irr_grid[:,0]
    y = irr_grid[:,1]
    z = irr_grid[:,2]
    
    ## GET NEW GRID FROM BOX SIZE
    if len(xrange) < 1:
        xrange = [ x.min(), x.max() ]
    if len(yrange) < 1:
        yrange = [ y.min(), y.max() ]

    ## ADD PBC PERIODIC
    if periodic is True:
        ## ADD PLUS X PLUS Y
        x = np.hstack(( x, irr_grid[:,0] + xrange[1] ))
        y = np.hstack(( y, irr_grid[:,1] + yrange[1] ))
        z = np.hstack(( z, irr_grid[:,2] ))

        ## ADD PLUS X
        x = np.hstack(( x, irr_grid[:,0] + xrange[1] ))
        y = np.hstack(( y, irr_grid[:,1] ))
        z = np.hstack(( z, irr_grid[:,2] ))

        ## ADD PLUS X MINUS Y
        x = np.hstack(( x, irr_grid[:,0] + xrange[1] ))
        y = np.hstack(( y, irr_grid[:,1] - yrange[1] ))
        z = np.hstack(( z, irr_grid[:,2] ))

        ## ADD PLUS Y
        x = np.hstack(( x, irr_grid[:,0] ))
        y = np.hstack(( y, irr_grid[:,1] + yrange[1] ))
        z = np.hstack(( z, irr_grid[:,2] ))

        ## ADD MINUS Y
        x = np.hstack(( x, irr_grid[:,0] ))
        y = np.hstack(( y, irr_grid[:,1] - yrange[1] ))
        z = np.hstack(( z, irr_grid[:,2] ))

        ## ADD MINUS X PLUS Y
        x = np.hstack(( x, irr_grid[:,0] - xrange[1] ))
        y = np.hstack(( y, irr_grid[:,1] + yrange[1] ))
        z = np.hstack(( z, irr_grid[:,2] ))

        ## ADD MINUS X
        x = np.hstack(( x, irr_grid[:,0] - xrange[1] ))
        y = np.hstack(( y, irr_grid[:,1] ))
        z = np.hstack(( z, irr_grid[:,2] ))

        ## ADD MINUS X MINUS Y
        x = np.hstack(( x, irr_grid[:,0] - xrange[1] ))
        y = np.hstack(( y, irr_grid[:,1] - yrange[1] ))
        z = np.hstack(( z, irr_grid[:,2] ))

    xi = np.arange( xrange[0], xrange[1]-spacing, spacing )
    yi = np.arange( yrange[0], yrange[1]-spacing, spacing )
            
    ## INTERPOLATE POINTS ON A GRID
    Zi = griddata( ( x, y ), z, ( xi[None,:], yi[:,None] ), method = "linear" )

    ## COMBINE RESULTS
    Xi, Yi = np.meshgrid( xi, yi )
    fixed_grid = np.array([ Xi.flatten(), Yi.flatten(), Zi.flatten() ]).transpose()
    
    ## RETURN FIXED GRID
    return fixed_grid

## FUNCTION TO FIT EXPONENTIAL FUNCITON
def fit_exp( t, R ):
    '''Perform a linear fit on exponental data'''
    ## LINEARIZE R
    log_R = np.log(R)
    
    ## FIT LINEAR FUNCTION TO LOG(R) VS T
    p = np.polyfit( t, np.log(R), 1 )
    
    ## EVALUATE FIT
    R_fit = np.polyval( p, t )
    
    ## COMPUTE ERROR OF FIT
    MSE    = np.sum( ( log_R - R_fit )**2. ) / R.size
    SS_tot = np.sum( ( log_R - log_R.mean() )**2. )
    SS_res = np.sum( ( log_R - R_fit )**2. )
    R_sq   = 1 - SS_res / SS_tot
    
    ## PRINT FITTING ERROR
    print( "tau: %.2f" % ( -1. / p[0] ) )
    print( "RMSE: %.2f" % ( np.sqrt( MSE ) ) )
    print( "R^2: %.2f" % ( R_sq ) )
    
    ## RETURN RESULTS WITH STATISTICS
    return { "tau"  : -1. / p[0],
             "RMSE" : np.sqrt(MSE),
             "R2"   : R_sq }

## FUNCTION READING INPUT CSV    
def load_csv( path_csv ):
    """FUNCTION READS A CSV AND RETURN A NUMPY ARRAY"""
    ## OPEN CSV
    with open( path_csv, 'r' ) as file_data:
        data = file_data.readlines()
    
    ## REMOVE COMMENTED LINES
    data = [ line for line in data if '#' not in line ]

    ## REMOVE END OF LINE MARKER
    data = [ n.strip( '\n' ) for n in data ]
    
    ## SPLIT LINES INTO ELEMENTS
    data = [ n.split() for n in data if len(n) > 0 ]
    
    ## CONVERT TO NUMPY ARRAY
    data = np.array( [ [ float( n ) for n in line if n != '' ] for line in data ] )
    
    ## RETURN RESULTS
    return data

## FUNCTION WRITING OUTPUT CSV      
def write_csv( data, path_csv ):
    """FUNCTION WRITES A CSV FROM A NUMPY ARRAY"""
    ## PRINTING    
    print( "  CSV SAVED TO {}".format( path_csv ) )
    
    ## OPEN FILE AND MAKE WRITABLE
    outfile = open( path_csv, 'w+' )
    
    ## WRITE FILE LINE BY LINE
    for line in data:
        outfile.write( '{:0.6f} {:0.6f} {:0.6f}\n'.format( line[0], line[1], line[2] ) )
    
    ## CLOSE FILE
    outfile.close()

# ### FUNCTION TO CONVERT SAM_ANALYSIS OUTPUT TO A PANDAS DATAFRAME
# def obj2df( data, column_labels = [], row_labels = [] ):
#     """FUNCTION LOADS DATA AND CONVERTS TO PANDAS"""
#     ## CHECK IF COLUMN LABELS ARE INPUT
#     if len(column_labels) < 0:
#         column_labels =

