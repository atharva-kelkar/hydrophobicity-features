"""
trajectory.py
script contains methods used to load and manipulate trajectory data

CREATED ON: 09/25/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os
import copy
import numpy as np
import mdtraj as md

## IMPORTING ITP READING TOOL 
from sam_analysis.core.gromacs import extract_itp

##############################################################################
## FUNCTIONS AND CLASSES
##############################################################################
## FUNCTION TO FIND ITP FILES IN TARGET DIRECTORY
def find_itp_files( directory = '.', extension = 'itp' ):
    r'''
    '''
    itp_files = []
    extension = extension.lower()
    
    for dirpath, dirnames, files in os.walk( directory ):
        for name in files:
            if extension in name:
                itp_files.append( os.path.join( dirpath, name ) )
    
    return itp_files

## FUNCTION TO ADD BONDS TO TRAJECTORY FROM ITP DATA
def add_bonds_to_traj( traj, itp_data = [] ):
    r'''
    Function takes data from itp files to add bonds to trajectory.topology object
    '''
    if not itp_data:
        print( "no itp data loaded! Load itp data." )
    else:
        for itp in itp_data:
            ## DOUBLE CHECK THAT MDTRAJ LOADS IN CORRECT RESIDUE
            res_atoms = [ [ atom for atom in residue.atoms ] for residue in traj.topology.residues if residue.name == itp.residue_name ]
            for res in res_atoms:
                for a1, a2 in itp.bonds:
                    traj.topology.add_bond( res[a1-1], res[a2-1] )

    return traj

## FUNCTION TO ADD BONDS TO WATER MOLECULES
def add_water_bonds( traj ):
    r'''
    Function to add water bonds to traj
    '''
    ## GATHER WATER GROUPS
    res_atoms = [ [ atom for atom in residue.atoms ] for residue in traj.topology.residues if residue.name in [ "SOL", "HOH" ] ]
    ## LOOP THROUGH WATER GROUPS
    for res in res_atoms:
        h1 = 0
        ## LOOP THROUGH MOLECULE
        for atom in res:
            if atom.element.symbol == "O":
                oxygen = atom
            elif atom.element.symbol == "H" and h1 < 1:
                hydrogen1 = atom
                h1 = 1
            elif atom.element.symbol == "H" and h1 > 0:
                hydrogen2 = atom
        ## BIND WATER
        traj.topology.add_bond( oxygen, hydrogen1 )
        traj.topology.add_bond( oxygen, hydrogen2 )
            
    return traj

## FUNCTION TO GATHER BOND INFO FROM ITP FILES
def gather_itp_data( path_data, verbose = False ):
    r'''
    Function reads itp files and loads itp data
    '''        
    ## ITP FILES IN WD/LOOKS FOR THOSE CREATED BY MDBuilder
    itp_files = []
    all_itp_files = find_itp_files( directory = path_data, extension = 'itp' )
    for itp in all_itp_files:
        with open( itp ) as f:
            line = f.readline()
            if "MDBuilder" in line:
                itp_files.append(itp)
    del all_itp_files
    
    ## EXTRACT ITP DATA
    itp_data = [ extract_itp( itp_file, verbose ) for itp_file in itp_files if itp_file != '' ]
    
    ## RETURN ITP DATA
    return itp_data    

## FUNCTION TO LOAD TRAJECTORY
def load_md_traj( path_traj,
                  input_prefix,
                  make_whole              = True,
                  add_bonds_from_itp      = True,
                  add_water_bonds_to_traj = True,
                  remove_dummy_atoms      = True,
                  standard_names          = False,
                  verbose                 = False ):
    """Function to load md trajectory (gromacs only)"""     
    ## FILE PATHS
    full_xtc = os.path.join( path_traj, input_prefix + ".xtc" )
    full_pdb = os.path.join( path_traj, input_prefix + ".pdb" )
    full_arc = os.path.join( path_traj, input_prefix + ".arc" )
    
    ## LOAD FULL TRAJECTORY
    if verbose is True:
        print( "  LOADING TRAJECTORY..." )
    
    ## CHECK IF XTC
    if os.path.exists( full_xtc ) is True:
        traj = md.load( full_xtc,
                        top = full_pdb,
                        standard_names = standard_names )
        
        ## UPDATE TRAJ
        traj = update_traj( traj,
                            path_traj,
                            make_whole              = make_whole,
                            add_bonds_from_itp      = add_bonds_from_itp,
                            add_water_bonds_to_traj = add_water_bonds_to_traj,
                            remove_dummy_atoms      = remove_dummy_atoms,
                            verbose                 = verbose )
    else:
        traj = md.load( full_arc, )
        traj_top = md.load( full_pdb,
                            standard_names = standard_names )
        traj.topology = copy.deepcopy( traj_top.topology )

    if verbose is True:
        print( "  TRAJECTORY LOADED SUCCESSFULLY!" )

    if verbose is True:
        ## PRINT TRAJ INFO
        traj_info( traj )
    
    return traj

## FUNCTION TO LOAD TRAJECTORY
def load_md_traj_frame( path_traj,
                        input_prefix,
                        make_whole              = True,
                        add_bonds_from_itp      = True,
                        add_water_bonds_to_traj = True,
                        remove_dummy_atoms      = True,
                        standard_names          = False,
                        verbose                 = False ):
    """Function to load md trajectory frame (gromacs only)"""     
    ## FILE PATHS
    full_pdb = os.path.join( path_traj, input_prefix + ".pdb" )
    
    ## LOAD FULL TRAJECTORY
    if verbose is True:
        print( "  LOADING TRAJECTORY..." )
    traj = md.load( full_pdb,
                    standard_names = standard_names )

    if verbose is True:
        print( "  TRAJECTORY LOADED SUCCESSFULLY!" )
    
    ## UPDATE TRAJ
    traj = update_traj( traj,
                        path_traj,
                        make_whole              = make_whole,
                        add_bonds_from_itp      = add_bonds_from_itp,
                        add_water_bonds_to_traj = add_water_bonds_to_traj,
                        remove_dummy_atoms      = remove_dummy_atoms,
                        verbose                 = verbose )

    if verbose is True:
        ## PRINT TRAJ INFO
        traj_info( traj )
    
    return traj

## FUNCTION TO LOAD TRAJECTORY
def iterload_md_traj( path_traj,
                      input_prefix,
                      iter_size               = 100,
                      make_whole              = True,
                      add_bonds_from_itp      = True,
                      add_water_bonds_to_traj = True,
                      remove_dummy_atoms      = True,
                      standard_names          = False,
                      verbose                 = False ):
    """Function to iteratively load md trajectory (gromacs only)"""
    ## LOAD FULL TRAJECTORY (STANDARD NAMES NOT AN OPTION IN ITERLOAD)
    full_xtc = os.path.join( path_traj, input_prefix + ".xtc" )
    full_pdb = os.path.join( path_traj, input_prefix + ".pdb" )
    traj = md.load( full_pdb,
                    standard_names = standard_names )  
    
    ## STORE TOPOLOGY
    traj_top = traj.topology
    
    ## ITERLOAD TRAJECTORY UPDATING TOPOLOGY EACH TIME
    for traj_chunk in md.iterload( full_xtc, top = full_pdb, chunk = iter_size ):
        ## UPDATE TOPOLOGY
        traj_chunk.topology = copy.deepcopy( traj_top )
        
        # UPDATE TRAJ
        traj_chunk = update_traj( traj_chunk,
                                  path_traj,
                                  make_whole              = make_whole,
                                  add_bonds_from_itp      = add_bonds_from_itp,
                                  add_water_bonds_to_traj = add_water_bonds_to_traj,
                                  remove_dummy_atoms      = remove_dummy_atoms,
                                  verbose                 = verbose )        
        ## YIELD CHUNK
        yield traj_chunk
    
## UPDATE TRAJ
def update_traj( traj,
                 path_traj,
                 make_whole              = True,
                 add_bonds_from_itp      = True,
                 add_water_bonds_to_traj = True,
                 remove_dummy_atoms      = True,
                 verbose                 = False ):
    """Function updates trajectory topology to add bonds and remove dummy atoms"""
    ## UPDATE TRAJECTORY
    if add_bonds_from_itp is True:
        ## GATHER ITP FILE DATA
        itp_data = gather_itp_data( path_traj, verbose = verbose )
        if verbose is True:
            print( " --- ADDED BONDS FROM ITP DATA ---" )
        
        ## UPDATE TRAJ WITH ITP DATA/ADDS BONDS
        traj = add_bonds_to_traj( traj, itp_data )

    if add_water_bonds_to_traj is True:
        traj = add_water_bonds( traj )
        if verbose is True:
            print( " --- ADDED BONDS TO WATER MOLECULES ---" )

    if remove_dummy_atoms is True:
        ## IDENTIFY DUMMY ATOM INDICES
        atom_indices = np.array([ atom.index for atom in traj.topology.atoms
                                  if atom.element.symbol != "VS" ])
        if verbose is True:
            print( " --- REMOVED DUMMY ATOMS FROM TRAJECTORY ---" )

        ## REDUCE TRAJECTORY BY REMOVING DUMMY ATOMS (EX. VIRTUAL SITE OF TIP4P)
        traj.atom_slice( atom_indices, inplace = True )
    
    ## RETURN UPDATED TRAJ
    return traj

## FUNCTION TO PRINT TRAJ INFO
def traj_info( traj ):
    """GET AND PRINT TRAJ INFORMATION"""
    print( "\n---  TRAJECTORY INFO ---" )
    print( "%-12s %6d" % ( "# Frames:", traj.n_frames ) )
    list_of_residues = set( [ residue.name for residue in traj.topology.residues ] )
    print( "%-12s %6d" % ( "# Residues:", len(list_of_residues) ) )
    print( "%-12s %6d" % ( "# Atoms:", len( list(traj.topology.atoms) ) ) )
    for residue in list_of_residues:
        n_residue = len([ ii for ii in traj.topology.residues if ii.name == residue ])
        print( "%-12s %6d" % ( "# " + residue + ":", n_residue ) )

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
    sam_dir = r"mixed_polar_sam"
    
    ## WORKING DIR
    working_dir = os.path.join(test_dir, sam_dir)
    
    ## LOAD TRAJECTORY
    path_traj = check_server_path( working_dir )

    ## LOAD TRAJECTORY
    input_prefix = "sam_prod"
    traj =  load_md_traj( path_traj,
                          input_prefix,
                          make_whole = True,
                          add_bonds_from_itp = True,
                          add_water_bonds_to_traj = True,
                          remove_dummy_atoms = True,
                          standard_names = False )
    

            