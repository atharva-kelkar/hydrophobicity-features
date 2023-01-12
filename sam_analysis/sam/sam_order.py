"""
sam_order.py 
script contains various methods to compute sam order

CREATED ON: 10/22/2020

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

## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORT MISC TOOLS
from sam_analysis.core.misc_tools import compute_com, ligand_heavy_atom_indices, \
                                         dihedral_lists, end_group_atom_indices
## IMPORT ORDER PARAMETERS
from sam_analysis.sam.order_parameters import compute_dihedral_angles, compute_psi

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## CLASS COMPUTING HEXATIC ORDER
class SamOrder:
    """class object used to compute various order parameters of a SAM"""
    def __init__( self,
                  traj            = None,
                  sim_working_dir = None,
                  input_prefix    = None,
                  recompute_order = False,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.traj            = traj
        self.sim_working_dir = sim_working_dir
        self.input_prefix    = input_prefix
        self.recompute_order = recompute_order

        ## LOAD TRAJ IN NOT INPUT
        if self.traj is None:
            self.traj = load_md_traj( path_traj    = self.sim_working_dir,
                                      input_prefix = self.input_prefix )
            
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)
                
    ## METHOD COMPUTING LIGAND TILT ANGLE
    def tilt_angle( self, per_frame = False ):
        ## PATH TO PKL
        out_name = self.input_prefix + "_ligand_tilt_angles.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )
        
        ## LOAD DATA
        if self.recompute_order is True or \
         os.path.exists( path_pkl ) is not True:
            ## GET LIGAND INDICES LISTS
            ligand_indices = ligand_heavy_atom_indices( self.traj )
            
            ## GET DUMMY ATOM INDICES
            dummy_indices = np.array([ lig[-1]+1 for lig in ligand_indices ]) # should always be hydrogen
            
            pairs = []
            for ndx, indices in zip( dummy_indices, ligand_indices ):
                ## GET LIGAND COM COORDS AND ASSIGN TO A DUMMY ATOM
                ligand_com = compute_com( self.traj, indices )
                self.traj.xyz[ :, ndx, : ] = ligand_com
                
                ## APPEND DUMMY AND TAIL ATOM
                pairs.append([ ndx, indices[0] ])
                
            ## DETERMINE LIGAND VECTORS
            pairs = np.array( pairs )
            vectors = md.compute_displacements( self.traj, pairs, periodic = True )
            dist = np.sqrt( np.sum( vectors**2., axis = 2 ) )
            tilt_angles = np.arccos( np.abs(vectors[:,:,2] / dist ) )
            
            ## CONVERT TO DEGREES
            tilt_angles = np.rad2deg(tilt_angles)
            
            ## SAVE PKL
            save_pkl( tilt_angles, path_pkl )
        else:
            ## LOAD DATA
            tilt_angles = load_pkl( path_pkl )
                
        ## OUTPUT SURFACE AVERAGE PER FRAME
        if per_frame is True:
            return tilt_angles.mean( axis = 1 )
        
        ## OUTPUT SURFACE ENSEMBLE AVERAGE
        return tilt_angles.mean()
    
    ## METHOD COMPUTING TRANS DIHEDRAL FRACTION
    def trans_fraction( self, per_frame = False ):
        ## PATH TO PKL
        out_name = self.input_prefix + "_ligand_dihedral_angles.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )
        
        ## LOAD DATA
        if self.recompute_order is True or \
         os.path.exists( path_pkl ) is not True:
            ## GET END GROUP ATOM INDICES
            end_atom_indices = end_group_atom_indices( self.traj )
                
            ## GET LIGAND INDICES LISTS
            ligand_indices_full = ligand_heavy_atom_indices( self.traj )

            ## REMOVE FULL ENDGROUP FROM LIGAND INDICES
            ii = 0
            dihedrals = np.zeros( shape = ( self.traj.n_frames,
                                            len(ligand_indices_full) ) )
            for ligand, end_group in zip( ligand_indices_full, end_atom_indices ):
                ## CREATE NEW LIGAND WITH AVERAGE END GROUP
                new_ligand = []
                for ll in ligand:
                    if ll not in end_group:
                        new_ligand.append( ll )
                        
                ## UPDATE NEW LIGAND WITH SINGLE END GROUP ATOM
                new_ligand.append( end_group[0] )
                
                ## COMPUTE COM POSITION OF END GROUP
                self.traj.xyz[:,new_ligand[-1],:] = compute_com( self.traj, end_group )
                
                ## GET DIHEDRAL LIST
                dihedral_list = dihedral_lists( [new_ligand] )
            
                ## CALCULATE DIHEDRALS
                dihedrals[:,ii] = compute_dihedral_angles( self.traj, dihedral_list, 
                                                           periodic = True )
                ## UPDATE COUNTER
                ii += 1
        
            ## SAVE PKL
            save_pkl( dihedrals, path_pkl )
        else:
            ## LOAD DATA
            dihedrals = load_pkl( path_pkl )
            
        ## OUTPUT SURFACE AVERAGE PER FRAME
        if per_frame is True:
            return dihedrals.mean( axis = 1 )
        
        ## OUTPUT SURFACE ENSEMBLE AVERAGE
        return dihedrals.mean()
        
    ## METHOD COMPUTING HEXATIC ORDER FROM AVG END GROUP COM
    def hexatic( self, per_frame = False ):
        ## PATH TO PKL
        out_name = self.input_prefix + "_ligand_hexatic_order.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )
        
        ## LOAD DATA
        if self.recompute_order is True or \
         os.path.exists( path_pkl ) is not True:
            ## GET END GROUP ATOM INDICES
            end_atom_indices = end_group_atom_indices( self.traj )
            
            ## GET DUMMY ATOM INDICES
            dummy_indices = np.array([ lig[-1] for lig in end_atom_indices ]) # should always be hydrogen
        
            ## COMPUTE COM POSITION OF END GROUP
            for ii, indices in enumerate(end_atom_indices):
                self.traj.xyz[:,dummy_indices[ii],:] = compute_com( self.traj, indices )
            
            ## LOOP THROUGH ENDGROUPS
            hexatic_order = np.zeros( shape = ( self.traj.time.size, 
                                                len(dummy_indices) ) )
            for ii, ndx in enumerate(dummy_indices):
                ## COMPUTE PSI FOR POINT INDEX
                hexatic_order[:,ii] = compute_psi( self.traj, ndx, dummy_indices, 
                                                   cutoff = 0.7, periodic = True )
            ## SAVE PKL
            save_pkl( hexatic_order, path_pkl )
        else:
            ## LOAD DATA
            hexatic_order = load_pkl( path_pkl )
                
        ## OUTPUT SURFACE AVERAGE PER FRAME
        if per_frame is True:
            return hexatic_order.mean( axis = 1 )
        
        ## OUTPUT SURFACE ENSEMBLE AVERAGE
        return hexatic_order.mean()

    ## METHOD COMPUTING RMSD OF LIGANDS
    def rmsd( self, per_frame = False ):
        ## PATH TO PKL
        out_name = self.input_prefix + "_ligand_rmsd.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )
        
        ## LOAD DATA
        if self.recompute_order is True or \
         os.path.exists( path_pkl ) is not True:
            ## GET LIGAND INDICES LISTS
            ligand_indices = ligand_heavy_atom_indices( self.traj )
            
            ## CREATE REF TRAJ (MEAN ATOM POSITIONS)
            ref_traj = self.traj[0]
            ref_traj.xyz[0,...] = self.traj.xyz[:].mean(axis=0)
            
            ## LOOP THROUGH INPUT LIGAND BY LIGAND
            rmsd = np.zeros( shape = ( self.traj.time.size,
                                       len(ligand_indices) ) )
            for ndx, indices in enumerate( ligand_indices ):
                rmsd[:,ndx] = md.rmsd( self.traj, ref_traj, frame = 0, atom_indices = indices )

            ## SAVE PKL
            save_pkl( rmsd, path_pkl )
        else:
            ## LOAD DATA
            rmsd = load_pkl( path_pkl )
            
        ## OUTPUT SURFACE AVERAGE PER FRAME
        if per_frame is True:
            return rmsd.mean( axis = 1 )
        
        ## OUTPUT SURFACE ENSEMBLE AVERAGE
        return rmsd.mean()

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE CLASS
    order_obj = SamOrder( **kwargs )
    
    ## AVERAGE TILT ANGLES
    tilt_angles = order_obj.tilt_angle( per_frame = False )    
    ## PER FRAME
    tilt_angles_per_frame = order_obj.tilt_angle( per_frame = True )    
    ## AVERAGE TRANS FRACTION
    trans_fraction = order_obj.trans_fraction( per_frame = False )    
    ## PER FRAME
    trans_fraction_per_frame = order_obj.trans_fraction( per_frame = True )
    ## AVERAGE HEXATIC ORDER
    hexatic_order = order_obj.hexatic( per_frame = False )    
    ## PER FRAME
    hexatic_order_per_frame = order_obj.hexatic( per_frame = True )
    ## AVERAGE ROOT MEAN SQ DEVIATION
    rmsd = order_obj.rmsd( per_frame = False )    
    ## PER FRAME
    rmsd_per_frame = order_obj.rmsd( per_frame = True )
    
    ## RETURN RESULTS
    return { "tilt_angles"                       : tilt_angles,
             "tilt_angles_per_frame"             : tilt_angles_per_frame,
             "trans_dihedral_fraction"           : trans_fraction,
             "trans_dihedral_fraction_per_frame" : trans_fraction_per_frame,
             "hexatic_order"                     : hexatic_order,
             "hexatic_order_per_frame"           : hexatic_order_per_frame,
             "ligand_rmsd"                       : rmsd,
             "ligand_rmsd_per_frame"             : rmsd_per_frame }
    
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
#    sam_dir = r"ordered_ch3_sam"
    sam_dir = r"mixed_conh2_sam"
    
    ## WORKING DIR
    working_dir = os.path.join(test_dir, sam_dir)
    
    ## LOAD TRAJECTORY
    path_traj = check_server_path( working_dir )

    ## LOAD TRAJECTORY
    input_prefix = "sam_prod"

    ## LOAD TRAJECTORY
    traj = load_md_traj( path_traj    = path_traj,
                         input_prefix = input_prefix )

    ## INITIALIZE CLASS
    order_obj = SamOrder( traj            = traj,  
                          sim_working_dir = path_traj,
                          input_prefix    = input_prefix,
                          recompute_order = True )
    
    ## AVERAGE TILT ANGLES
    tilt_angles = order_obj.tilt_angle( per_frame = False )    
    ## PER FRAME
    tilt_angles_per_frame = order_obj.tilt_angle( per_frame = True )    
    ## AVERAGE TRANS FRACTION
    trans_fraction = order_obj.trans_fraction( per_frame = False )    
    ## PER FRAME
    trans_fraction_per_frame = order_obj.trans_fraction( per_frame = True )
    ## AVERAGE HEXATIC ORDER
    hexatic_order = order_obj.hexatic( per_frame = False )    
    ## PER FRAME
    hexatic_order_per_frame = order_obj.hexatic( per_frame = True )
    ## AVERAGE ROOT MEAN SQ DEVIATION
    rmsd = order_obj.rmsd( per_frame = False )    
    ## PER FRAME
    rmsd_per_frame = order_obj.rmsd( per_frame = True )
    
    ## PRINT OUT RESULTS
    print( "\nAVG TILT ANGLE {}".format( tilt_angles ) )
    print( "TILT ANGLE PER FRAME SIZE {}".format( tilt_angles_per_frame.shape ) )
    print( "AVG TRANS FRACTION {}".format( trans_fraction ) )
    print( "TRANS FRACTION PER FRAME SIZE {}".format( trans_fraction_per_frame.shape ) )
    print( "AVG HEXATIC ORDER {}".format( hexatic_order ) )
    print( "HEXATIC ORDER PER FRAME SIZE {}".format( hexatic_order_per_frame.shape ) )
    print( "AVG RMSD {}".format( rmsd ) )
    print( "RMSD PER FRAME SIZE {}".format( rmsd_per_frame.shape ) )
    
           