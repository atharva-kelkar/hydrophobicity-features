"""
embedded_water.py 
script contains functions to determine embedded water in a SAM

CREATED ON: 01/05/2021

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

## IMPORT HEIGHT FUNCTION
from sam_analysis.sam.sam_height import SamHeight
## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj
## IMPORT MISC TOOLS
from sam_analysis.core.misc_tools import compute_displacements

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## EMBEDDED WATER CLASS
class EmbeddedWater:
    """class object used to compute embedded water in a SAM"""
    def __init__( self,
                  traj                     = None,
                  sim_working_dir          = None,
                  input_prefix             = None,
                  embedded_ref             = "sam_height",
                  water_residues           = [ "SOL", "HOH" ],
                  periodic                 = True,
                  recompute_embedded_water = False,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.traj            = traj
        self.sim_working_dir = sim_working_dir
        self.input_prefix    = input_prefix
        self.water_residues  = water_residues
        self.periodic        = periodic
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)
        
        ## LOAD TRAJ IN NOT INPUT
        if self.traj is None:
            self.traj = load_md_traj( path_traj    = sim_working_dir,
                                      input_prefix = input_prefix )
        ## COMPUTE SAM HEIGHT
        if type(embedded_ref) is str:
            ## MAKE STRING LOWER CASE
            embedded_ref = embedded_ref.lower()
            
            ## INITIALIZE CLASS
            height_obj = SamHeight( **self.__dict__ )
            
            ## COMPUTE REFERENCE FROM SAM HEIGHT
            if embedded_ref == "willard_chandler":
                ## WILLARD-CHANDLER INTERFACE
                self.embedded_ref = height_obj.willard_chandler( per_frame = False )
            ## COMPUTE REFERENCE FROM END GROUP HEIGHT
            elif embedded_ref == "sam_height":
                ## AVERAGE END GROUP HEIGHT
                self.embedded_ref = height_obj.average_end_group( per_frame = False )
                
        ## COMPUTE REFERENCE FROM INPUT
        else:
            self.embedded_ref = embedded_ref
                
    ## METHOD TO COUNT EMBEDDED WATERS
    def count( self, per_frame = False ):
        ## GET WATER INDICES
        water_indices = np.array([ [ atom.index for atom in residue.atoms ] 
                                     for residue in self.traj.topology.residues
                                     if residue.name in self.water_residues ] )
        
        ## GET WATER OXYGEN INDICES
        oxygen_indices = np.array([ water[0] for water in water_indices ])

        ## COMPUTE DISTANCE TO Z_REF
        ref_coords = [ self.traj.unitcell_lengths[0,0], 
                       self.traj.unitcell_lengths[0,1], 
                       self.embedded_ref ]
        z_dist = compute_displacements( self.traj,
                                        atom_indices   = oxygen_indices,
                                        box_dimensions = self.traj.unitcell_lengths,
                                        ref_coords     = ref_coords,
                                        periodic       = self.periodic )[...,2]
        
        ## REDUCE ATOMS TO THOSE INSIDE CUTOFFS
        mask = np.logical_and( z_dist > -0.5, z_dist < 0.0 )        
        
        ## COMPUTE NUMBER BELOW INTERFACE
        embedded_waters = np.sum( mask, axis = 1 )
        
        ## PER FRAME
        if per_frame is False:
            embedded_waters = embedded_waters.mean()
            
        ## RETURN RESULTS
        return embedded_waters

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## EMBEDDED WATERS
    embed_obj = EmbeddedWater( **kwargs )
    
    ## COMPUTE NUMBER EMBEDDED WATERS
    embedded_waters = embed_obj.count( per_frame = False )
    
    ## RETURN RESULTS
    return { "embedded_waters" : embedded_waters, }

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
        
    ## EMBEDDED WATERS
    embed_obj = EmbeddedWater( traj                     = traj,
                               sim_working_dir          = path_traj,
                               input_prefix             = input_prefix,
                               embedded_ref             = "sam_height",
                               water_residues           = [ "SOL", "HOH" ],
                               periodic                 = True,
                               recompute_embedded_water = False )
    
    ## COMPUTE NUMBER EMBEDDED WATERS
    embedded_waters = embed_obj.count( per_frame = False )
    
    ## COMPUTE NUMBER EMBEDDED WATERS PER FRAME
    embedded_waters_per_frame = embed_obj.count( per_frame = True )
    
    ## PRINT OUT RESULTS
    print( "AVG NUMBER EMBEDDED WATERS: {}".format( embedded_waters ) )
    print( "EMBEDDED WATERS PER FRAME SIZE: {}".format( embedded_waters_per_frame.shape ) )
        