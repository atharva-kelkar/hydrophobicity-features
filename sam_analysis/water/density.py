"""
density.py 
script contains functions to compute density profiles in a MD trajectory

CREATED ON: 12/02/2020

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
## IMPORT COPY
import copy
## IMPORT NUMPY
import numpy as np

## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj
## IMPORT HEIGHT FUNCTION
from sam_analysis.sam.sam_height import SamHeight
## IMPORT MISC TOOLS
from sam_analysis.core.misc_tools import compute_displacements, compute_com, \
                                         end_group_atom_indices

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## CLASS TO COMPUTE DENSITY PROFILE
class Density:
    """class object used to compute density profiles"""
    def __init__( self,
                  traj              = None,
                  sim_working_dir   = None,
                  input_prefix      = None,
                  z_ref             = "willard_chandler",
                  z_range           = ( -2, 2. ),
                  z_bin_width       = 0.005,
                  water_residues    = [ "SOL", "HOH" ],
                  periodic          = True,
                  recompute_density = False,
                  verbose           = True,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.traj              = traj
        self.sim_working_dir   = sim_working_dir
        self.input_prefix      = input_prefix
        self.z_ref             = z_ref
        self.range             = z_range
        self.bin_width         = z_bin_width
        self.water_residues    = water_residues
        self.periodic          = periodic
        self.recompute_density = recompute_density
        self.verbose           = verbose
        
        ## SET UP PROFILE 
        self.n_bins = np.floor( np.abs( z_range[-1] - z_range[0] ) / z_bin_width ).astype("int")
        self.z      = np.arange( z_range[0] + 0.5*z_bin_width, 
                                 z_range[-1] + 0.5*z_bin_width, 
                                 z_bin_width )
        
        ## LOAD TRAJ IN NOT INPUT
        if self.traj is None:
            self.traj = load_md_traj( path_traj    = self.sim_working_dir,
                                      input_prefix = self.input_prefix )

        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)
        
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
        
    ## INSTANCE COMPUTING DENSITY
    def rho( self, traj, atom_indices ):
        """Function computes density profile"""        
        ## COMPUTE Z DISTANCE FROM Z_REF
        ref_coords = [ 0.5*traj.unitcell_lengths[0,0],
                       0.5*traj.unitcell_lengths[0,1],
                       self.z_ref ]
        z_dist = compute_displacements( traj,
                                        atom_indices   = atom_indices,
                                        box_dimensions = traj.unitcell_lengths,
                                        ref_coords     = ref_coords,
                                        periodic       = self.periodic )[...,2]
        
        ## BIN ATOM POSITIONS
        histo = np.histogram( z_dist, bins = self.n_bins, range = self.range, )[0]
                
        ## COMPUTE VOLUME OF SLICE
        volume = self.bin_width * traj.unitcell_lengths[0,0] * traj.unitcell_lengths[0,1]
        
        ## NORMALIZE PROFILES BY VOLUME AND FRAMES
        rho = histo / volume / traj.n_frames
                
        ## RETURN RESULTS
        return rho
        
    ## INSTANCE COMPUTING WATER DENSITY
    def water( self, com = True ):
        """Function computes water density profile"""
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
        ## CHECK IF EMPTY
        if atom_indices.size > 0:
            rho = self.rho( traj, atom_indices )
        else:
            rho = np.zeros_like( self.z )
            
        ## RETURN RESULTS
        return rho
        
    ## INSTANCE COMPUTING END GROUP DENSITY
    def end_group( self ):
        """Function computes end group density profile"""
        ## CREATE COPY OF TRAJ
        traj = copy.deepcopy(self.traj)
        
        ## GET ATOM GROUPS TO COMPUTE DENSITY
        residue_group_indices = end_group_atom_indices( traj, labels = False )    
        atom_indices = np.array([ residue[0] for residue in residue_group_indices ])
                
        ## UPDATE TRAJ SO HEAVY ATOM HAS COM POSITION
        for ndx, res_indices in zip( atom_indices, residue_group_indices ):
            traj.xyz[:,ndx,:] = compute_com( traj, res_indices )
            
        ## CHECK IF EMPTY
        if atom_indices.size > 0:
            rho = self.rho( traj, atom_indices )
        else:
            rho = np.zeros_like( self.z )
            
        ## RETURN RESULTS
        return rho
    
    ## INSTANCE COMPUTING END GROUP OXYGEN DENSITY
    def end_group_oxygen( self ):
        """Function computes end group oxygen density profile"""        
        ## GET ATOM GROUPS TO COMPUTE DENSITY
        residue_group_indices = end_group_atom_indices( self.traj, labels = False )    
        
        ## EXCLUDE END GROUP
        end_group_indices = []
        for indices in residue_group_indices:
            end_group_indices += indices

        ## EXTRACT OXYGEN INDICES IN RESIDUE LIST
        atom_indices = np.array([ atom.index for atom in self.traj.topology.atoms 
                                   if atom.residue.name not in self.water_residues
                                   and atom.index in end_group_indices
                                   and atom.element.symbol == "O" ])
        ## CHECK IF EMPTY
        if atom_indices.size > 0:
            rho = self.rho( self.traj, atom_indices )
        else:
            rho = np.zeros_like( self.z )
            
        ## RETURN RESULTS
        return rho
    
    ## INSTANCE COMPUTING END GROUP NITROGEN DENSITY
    def end_group_nitrogen( self ):
        """Function computes end group nitrogen density profile"""        
        ## GET ATOM GROUPS TO COMPUTE DENSITY
        residue_group_indices = end_group_atom_indices( self.traj, labels = False )    
        
        ## EXCLUDE END GROUP
        end_group_indices = []
        for indices in residue_group_indices:
            end_group_indices += indices

        ## EXTRACT CARBON INDICES IN RESIDUE LIST
        atom_indices = np.array([ atom.index for atom in self.traj.topology.atoms 
                                   if atom.residue.name not in self.water_residues
                                   and atom.index in end_group_indices
                                   and atom.element.symbol == "N" ])
        ## CHECK IF EMPTY
        if atom_indices.size > 0:
            rho = self.rho( self.traj, atom_indices )
        else:
            rho = np.zeros_like( self.z )
            
        ## RETURN RESULTS
        return rho
        
    ## INSTANCE COMPUTING BACKBONE CARBON DENSITY
    def backbone_carbon( self, include_end_group = False ):
        """Function computes backbone carbon density profile"""        
        ## GET ATOM GROUPS TO COMPUTE DENSITY
        residue_group_indices = end_group_atom_indices( self.traj, labels = False )    
        
        end_group_indices = []
        if include_end_group is False:
            ## EXCLUDE END GROUP
            for indices in residue_group_indices:
                end_group_indices += indices

        ## EXTRACT CARBON INDICES IN RESIDUE LIST
        atom_indices = np.array([ atom.index for atom in self.traj.topology.atoms 
                                   if atom.residue.name not in self.water_residues
                                   and atom.index not in end_group_indices
                                   and atom.element.symbol == "C" ])
        ## CHECK IF EMPTY
        if atom_indices.size > 0:
            rho = self.rho( self.traj, atom_indices )
        else:
            rho = np.zeros_like( self.z )
            
        ## RETURN RESULTS
        return rho
        
    ## INSTANCE COMPUTING BACKBONE HYDROGEN DENSITY
    def backbone_hydrogen( self, include_end_group = False ):
        """Function computes backbone hydrogen density profile"""        
        ## GET ATOM GROUPS TO COMPUTE DENSITY
        residue_group_indices = end_group_atom_indices( self.traj, labels = False )    
        
        end_group_indices = []
        if include_end_group is False:
            ## EXCLUDE END GROUP
            for indices in residue_group_indices:
                end_group_indices += indices

        ## EXTRACT HYDROGEN INDICES IN RESIDUE LIST
        atom_indices = np.array([ atom.index for atom in self.traj.topology.atoms 
                                   if atom.residue.name not in self.water_residues
                                   and atom.index not in end_group_indices
                                   and atom.element.symbol == "H" ])
        ## CHECK IF EMPTY
        if atom_indices.size > 0:
            rho = self.rho( self.traj, atom_indices )
        else:
            rho = np.zeros_like( self.z )
            
        ## RETURN RESULTS
        return rho
        
    ## INSTANCE COMPUTING BACKBONE HYDROGEN DENSITY
    def backbone_sulfur( self, include_end_group = False ):
        """Function computes backbone hydrogen density profile"""        
        ## GET ATOM GROUPS TO COMPUTE DENSITY
        residue_group_indices = end_group_atom_indices( self.traj, labels = False )    
        
        end_group_indices = []
        if include_end_group is False:
            ## EXCLUDE END GROUP
            for indices in residue_group_indices:
                end_group_indices += indices

        ## EXTRACT SULFUR INDICES IN RESIDUE LIST
        atom_indices = np.array([ atom.index for atom in self.traj.topology.atoms 
                                   if atom.residue.name not in self.water_residues
                                   and atom.index not in end_group_indices
                                   and atom.element.symbol == "S" ])
        ## CHECK IF EMPTY
        if atom_indices.size > 0:
            rho = self.rho( self.traj, atom_indices )
        else:
            rho = np.zeros_like( self.z )
            
        ## RETURN RESULTS
        return rho

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE CLASS
    density_obj = Density( **kwargs )
    
    ## COMPUTE WATER DENSITY PROFILE
    water_density = density_obj.water( com = True )
    
    ## COMPUTE END GROUP DENSITY PROFILE
    end_group_density = density_obj.end_group()

    ## COMPUTE END GROUP OXYGEN DENSITY PROFILE
    end_group_oxygen_density = density_obj.end_group_oxygen()

    ## COMPUTE END GROUP NITROGEN DENSITY PROFILE
    end_group_nitrogen_density = density_obj.end_group_nitrogen()

    ## COMPUTE BACKBONE CARBON DENSITY PROFILE
    backbone_carbon_density = density_obj.backbone_carbon( include_end_group = False )

    ## COMPUTE BACKBONE HYDROGEN DENSITY PROFILE
    backbone_hydrogen_density = density_obj.backbone_hydrogen( include_end_group = False )

    ## COMPUTE BACKBONE SULFUR DENSITY PROFILE
    backbone_sulfur_density = density_obj.backbone_sulfur( include_end_group = False )
    
    ## RETURN RESULTS
    return { "density_z"                  : density_obj.z,
             "water_density"              : water_density,
             "end_group_density"          : end_group_density,
             "end_group_oxygen_density"   : end_group_oxygen_density,
             "end_group_nitrogen_density" : end_group_nitrogen_density,
             "backbone_carbon_density"    : backbone_carbon_density,
             "backbone_hydrogen_density"  : backbone_hydrogen_density,
             "backbone_sulfur_density"    : backbone_sulfur_density }
        
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
    
    ## DENSITY OBJECT
    density_obj = Density( traj              = traj,
                           sim_working_dir   = path_traj,
                           input_prefix      = input_prefix,
                           z_ref             = "willard_chandler",
                           water_residues    = [ "SOL", "HOH" ],
                           periodic          = True,
                           recompute_density = True,
                           verbose           = True )
    
    ## COMPUTE WATER DENSITY PROFILE
    water_density = density_obj.water( com = True )
    
    ## COMPUTE END GROUP DENSITY PROFILE
    end_group_density = density_obj.end_group()

    ## COMPUTE END GROUP OXYGEN DENSITY PROFILE
    end_group_oxygen_density = density_obj.end_group_oxygen()

    ## COMPUTE END GROUP NITROGEN DENSITY PROFILE
    end_group_nitrogen_density = density_obj.end_group_nitrogen()

    ## COMPUTE BACKBONE CARBON DENSITY PROFILE
    backbone_carbon_density = density_obj.backbone_carbon( include_end_group = False )

    ## COMPUTE BACKBONE HYDROGEN DENSITY PROFILE
    backbone_hydrogen_density = density_obj.backbone_hydrogen( include_end_group = False )

    ## COMPUTE BACKBONE SULFUR DENSITY PROFILE
    backbone_sulfur_density = density_obj.backbone_sulfur( include_end_group = False )
    
    ## PRINT OUT RESULTS
    print( "WATER DENSITY PROFILE: {}\n".format( water_density.shape ) )
    print( "END GROUP DENSITY PROFILE: {}\n".format( end_group_density.shape ) )
    print( "END GROUP OXYGEN DENSITY PROFILE: {}\n".format( end_group_oxygen_density.shape ) )
    print( "END GROUP NITROGEN DENSITY PROFILE: {}\n".format( end_group_nitrogen_density.shape ) )
    print( "BACKBONE CARBON DENSITY PROFILE: {}\n".format( backbone_carbon_density.shape ) )
    print( "BACKBONE HYDROGEN DENSITY PROFILE: {}\n".format( backbone_hydrogen_density.shape ) )
    print( "BACKBONE SULFUR DENSITY PROFILE: {}\n".format( backbone_sulfur_density.shape ) )
    