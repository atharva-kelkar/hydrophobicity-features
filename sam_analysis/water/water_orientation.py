"""
water_orientation.py 
script contains functions to compute water molecule orientation in a MD trajectory

CREATED ON: 12/01/2020

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
## IMPORT MDTRAJ
import mdtraj as md

## IMPORT HEIGHT FUNCTION
from sam_analysis.sam.sam_height import SamHeight
## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORT MISC TOOLS
from sam_analysis.core.misc_tools import compute_displacements

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## CLASS TO COMPUTE WATER ORIENATION
class WaterOrientation:
    """class object used to compute water orientation"""
    def __init__( self,
                  traj                  = None,
                  sim_working_dir       = None,
                  input_prefix          = None,
                  z_ref                 = "willard_chandler",
                  z_cutoff              = 0.3,
                  phi_range             = ( 0, 180. ),
                  phi_bin_width         = 10.,
                  water_residues        = [ "SOL", "HOH" ],
                  periodic              = True,
                  recompute_orientation = False,
                  verbose               = True, 
                  print_freq            = 100,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.traj                  = traj
        self.sim_working_dir       = sim_working_dir
        self.input_prefix          = input_prefix
        self.z_ref                 = z_ref
        self.z_cutoff              = z_cutoff
        self.range                 = phi_range
        self.bin_width             = phi_bin_width
        self.water_residues        = water_residues
        self.periodic              = periodic
        self.recompute_orientation = recompute_orientation
        self.verbose               = verbose
        self.print_freq            = print_freq
        
        ## SET UP DISTRIBUTION
        self.n_bins = np.floor( phi_range[-1] / phi_bin_width ).astype("int")
        self.phi    = np.arange( phi_range[0] + 0.5*phi_bin_width, 
                                 phi_range[-1] + 0.5*phi_bin_width, 
                                 phi_bin_width )
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

        ## LOAD TRAJ IN NOT INPUT
        if self.traj is None:
            self.traj = load_md_traj( path_traj    = self.sim_working_dir,
                                      input_prefix = self.input_prefix ) 
                
        ## COMPUTE SAM HEIGHT
        if type(z_ref) is str:
            z_ref = z_ref.lower()
            
            ## INITIALIZE CLASS
            height_obj = SamHeight( **self.__dict__ )
            
            ## COMPUTE REFERENCE FROM SAM HEIGHT
            if z_ref == "willard_chandler":            
                ## WILLARD-CHANDLER INTERFACE
                self.z_ref = height_obj.willard_chandler( per_frame = False )
                
            elif z_ref == "sam_height":
                ## AVERAGE END GROUP HEIGHT
                self.z_ref = height_obj.average_end_group( per_frame = False ) 

        ## METHODS
        self._methods = { "dipole"  : self.dipole(),
                          "oh_bond" : self.oh_bond() }
                
    def __str__( self ):
        return "Water Molecule Orientation"
    
    ## INSTANCE COMPUTING DIPOLE ORIENTATION
    def dipole( self ):
        """COMPUTES DIPOLE ANGLE OF WATERS (ASSUMES MOLECULES ARE CONNECTED I.E.
        NO PBC)"""
        ## PATH TO PKL
        out_name = self.input_prefix + "_dipole_orientations.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )

        ## LOAD DATA
        if self.recompute_orientation is True or \
         os.path.exists( path_pkl ) is not True:
            ## GET WATER INDICES
            water_indices = np.array([ [ atom.index for atom in residue.atoms ] 
                                         for residue in self.traj.topology.residues
                                         if residue.name in self.water_residues ] )
            
            ## GET WATER OXYGEN INDICES
            oxygen_indices = np.array([ water[0] for water in water_indices ])
            
            ## GET WATER HYDROGEN INDICES
            hydrogen1_indices = np.array([ water[1] for water in water_indices ])
            hydrogen2_indices = np.array([ water[2] for water in water_indices ])
                    
            ## COMPUTE DISTANCE TO Z_REF
            ref_coords = [ self.traj.unitcell_lengths[0,0], 
                           self.traj.unitcell_lengths[0,1], 
                           self.z_ref ]
            z_dist = compute_displacements( self.traj,
                                            atom_indices   = oxygen_indices,
                                            box_dimensions = self.traj.unitcell_lengths,
                                            ref_coords     = ref_coords,
                                            periodic       = self.periodic )[...,2]
        
            ## REDUCE ATOMS TO THOSE INSIDE CUTOFFS
            mask = np.logical_and( z_dist > -(self.z_cutoff + 0.5), # always includes head groups
                                   z_dist < self.z_cutoff )
            
            ## ATOM POSITIONS
            oxygen_positions    = self.traj.xyz[:,oxygen_indices,:]
            hydrogen1_positions = self.traj.xyz[:,hydrogen1_indices,:]
            hydrogen2_positions = self.traj.xyz[:,hydrogen2_indices,:]
            
            ## AVERAGE HYDROGEN POSITION
            hydrogen_avg_positions = 0.5 * ( hydrogen1_positions + hydrogen2_positions )
            
            ## CALCULATE DIPOLE VECTOR
            dipole_vectors = oxygen_positions - hydrogen_avg_positions
            
            ## CALCULATE DIPOLE ANGLE
            phi = orientation_angle( dipole_vectors )
            
            ## STORE PHI, MASK, WATER_INDICES
            phi_dict = { "angles"  : phi,
                         "mask"    : mask,
                         "indices" : water_indices }

            ## SAVE PHI DICTIONARY TO PKL
            save_pkl( phi_dict, path_pkl )
        else:
            ## LOAD FROM FILE
            phi_dict = load_pkl( path_pkl )
        
        ## RETURN FILTERED PHI ARRAY
        return phi_dict["angles"][phi_dict["mask"]]

    ## INSTANCE COMPUTING OH BOND ORIENTATION
    def oh_bond( self ):
        """COMPUTES OH BOND ANGLE OF WATERS"""
        ## PATH TO PKL
        out_name = self.input_prefix + "_oh_bond_orientations.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )

        ## LOAD DATA
        if self.recompute_orientation is True or \
         os.path.exists( path_pkl ) is not True:
            ## GET WATER INDICES
            water_indices = np.array([ [ atom.index for atom in residue.atoms ] 
                                         for residue in self.traj.topology.residues
                                         if residue.name in self.water_residues ] )
            
            ## GET WATER OXYGEN INDICES
            oxygen_indices = np.array([ water[0] for water in water_indices ])
            
            ## GET WATER HYDROGEN INDICES
            hydrogen1_indices = np.array([ water[1] for water in water_indices ])
            hydrogen2_indices = np.array([ water[2] for water in water_indices ])
                    
            ## COMPUTE DISTANCE TO Z_REF
            ref_coords = [ self.traj.unitcell_lengths[0,0],
                           self.traj.unitcell_lengths[0,1],
                           self.z_ref ]
            z_dist = compute_displacements( self.traj,
                                            atom_indices   = oxygen_indices,
                                            box_dimensions = self.traj.unitcell_lengths,
                                            ref_coords     = ref_coords,
                                            periodic       = self.periodic )[...,2]
        
            ## REDUCE ATOMS TO THOSE INSIDE CUTOFFS
            mask = np.logical_and( z_dist > -(self.z_cutoff + 0.5),
                                   z_dist < self.z_cutoff )
                                    
            ## CALCULATE OH BOND1 VECTOR
            pairs = np.stack(( hydrogen1_indices, oxygen_indices )).transpose()
            oh_bond1_vectors = md.compute_displacements( self.traj, pairs, 
                                                         periodic = self.periodic )
            
            ## CALCULATE OH BOND1 ANGLE
            phi1 = orientation_angle( oh_bond1_vectors )
            
            ## CALCULATE OH BOND1 VECTOR
            pairs = np.stack(( hydrogen2_indices, oxygen_indices )).transpose()
            oh_bond2_vectors = md.compute_displacements( self.traj, pairs, 
                                                         periodic = self.periodic )
            
            ## CALCULATE OH BOND1 ANGLE
            phi2 = orientation_angle( oh_bond2_vectors )
            
            ## COMBINE RESULTS
            phi = np.stack(( phi1, phi2 )).transpose(( 1, 2, 0 ))
            
            ## STORE PHI, MASK, WATER_INDICES
            phi_dict = { "angles"  : phi,
                         "mask"    : mask,
                         "indices" : water_indices }

            ## SAVE PHI DICTIONARY TO PKL
            save_pkl( phi_dict, path_pkl )
        else:
            ## LOAD FROM FILE
            phi_dict = load_pkl( path_pkl )
        
        ## RETURN FILTERED PHI ARRAY
        return phi_dict["angles"][phi_dict["mask"],:].flatten()
    
    ## INSTANCE COMPUTING ANGLE DISTRIBUTION
    def distribution( self, angle_type ):
        """function to histogram angle distributions"""
        ## COMPUTE ANGLES
        angles = self._methods[angle_type]
        
        ## HISTOGRAM ANGLES
        dist = np.histogram( angles, bins = self.n_bins, range = self.range )[0]
        
        ## NORMALIZE HISTOGRAM
        norm_const = np.trapz( dist, dx = self.bin_width ) * np.sin( np.deg2rad(self.phi) )
        norm_dist  = dist / norm_const
        
        ## RETURN RESULTS
        return norm_dist

### FUNCTION TO COMPUTE ORIENTATION ANGLE OF WATER IN FRAME
def orientation_angle( vector1, vector2 = "normal" ):
    """function to compute angle of a vector"""
    ## CHECK IF VECTOR 2 IS NORMAL
    if vector2 == "normal":
        vector2 = np.array([ 0, 0, 1 ])
    
    ## CHECK DIMENSIONS OF DATA
    dim = len(vector1.shape)
    
    ## COMPUTE MAGNITUES
    mag1 = np.sqrt( np.sum( vector1**2, axis = dim-1 ) )
    mag2 = np.sqrt( np.sum( vector2**2 ) )
    
    ## COMPUTE PHI
    cos_phi = np.sum( vector1 * vector2, axis = dim-1 ) / mag1 / mag2
    
    # ADJUST FOR ROUNDING ERROR
    cos_phi = np.round( cos_phi, 4 )
    
    ## CONVERT TO DEGREES
    phi = np.rad2deg( np.arccos( cos_phi ) )
    
    ## RETURN RESULTS
    return phi

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE CLASS
    orient_obj = WaterOrientation( **kwargs )
    
    ## COMPUTE DIPOLE ANGLE DISTRIBUTION
    dipole_angle_dist = orient_obj.distribution( "dipole" )
    
    ## COMPUTE DIPOLE ANGLE DISTRIBUTION
    oh_bond_angle_dist = orient_obj.distribution( "oh_bond" )
    
    ## RETURN RESULTS
    return { "orientation_phi"            : orient_obj.phi,
             "dipole_angle_distribution"  : dipole_angle_dist,
             "oh_bond_angle_distribution" : oh_bond_angle_dist, }
        
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

    ## LOAD TRAJECTORY
    input_prefix = "sam_prod"
    
    ## LOAD TRAJ
    traj = load_md_traj( path_traj    = path_traj,
                         input_prefix = input_prefix )
        
    ## WATER ORIENTATION
    orient_obj = WaterOrientation( traj                  = traj,
                                   sim_working_dir       = path_traj,
                                   input_prefix          = input_prefix,
                                   z_ref                 = "willard_chandler",
                                   z_cutoff              = 0.3,
                                   phi_range             = ( 0, 180. ),
                                   phi_bin_width         = 10.,
                                   water_residues        = [ "SOL", "HOH" ],
                                   periodic              = True,
                                   recompute_orientation = True,
                                   verbose               = True, 
                                   print_freq            = 10, )    
    ## COMPUTE DIPOLE ANGLES
    dipole_angles = orient_obj.dipole()
    
    ## COMPUTE OH BOND ANGLES
    oh_bond_angles = orient_obj.oh_bond()
    
    ## COMPUTE DIPOLE ANGLE DISTRIBUTION
    dipole_angle_dist = orient_obj.distribution( "dipole" )
    
    ## COMPUTE DIPOLE ANGLE DISTRIBUTION
    oh_bond_angle_dist = orient_obj.distribution( "oh_bond" )
    
    ## PRINT OUT RESULTS
    print( "DIPOLE ANGLE ARRAY SIZE: {}".format( dipole_angles.shape ) )
    print( "OH BOND ANGLE ARRAY SIZE: {}".format( oh_bond_angles.shape ) )
    print( "DIPOLE ANGLE SIZE: {}".format( dipole_angle_dist.shape ) )
    print( "OH BOND ANGLE SIZE: {}".format( oh_bond_angle_dist.shape ) )
    