"""
h_bond.py 
script contains functions to compute hbonds in a MD trajectory

CREATED ON: 11/23/2020

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
## IMPORT SERIAL AND PARALLEL FUNCTIONS
from sam_analysis.core.parallel import run_parallel
## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj, iterload_md_traj
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORT MISC TOOLS
from sam_analysis.core.misc_tools import track_time, compute_displacements

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## CLASS TO COMPUTE HYDROGEN BOND TRIPLETS
class HydrogenBonds:
    """class object used to compute hydrogen bonds"""
    def __init__( self,
                  sim_working_dir          = None,
                  input_prefix             = None,
                  n_procs                  = 1,
                  iter_size                = 100,
                  z_ref                    = "willard_chandler", 
                  z_cutoff                 = 0.3, 
                  hbond_r_cutoff           = 0.35, 
                  hbond_phi_cutoff         = 0.523598,
                  periodic                 = True,
                  recompute_hbond_triplets = False,
                  verbose                  = True, 
                  print_freq               = 100,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.sim_working_dir          = sim_working_dir
        self.input_prefix             = input_prefix
        self.n_procs                  = n_procs
        self.iter_size                = iter_size
        self.z_ref                    = z_ref
        self.z_cutoff                 = z_cutoff
        self.hbond_r_cutoff           = hbond_r_cutoff
        self.hbond_phi_cutoff         = hbond_phi_cutoff
        self.periodic                 = periodic
        self.recompute_hbond_triplets = recompute_hbond_triplets
        self.verbose                  = verbose
        self.print_freq               = print_freq
        
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
                
    def __str__( self ):
        return "Hydrogen Bonds"
    
    ## INSTANCE COMPUTING TRIPLETS
    def triplets( self ):
        ## PATH TO PKL
        out_name = self.input_prefix + "_hbond_triplets.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )

        ## LOAD DATA
        if self.recompute_hbond_triplets is True or \
           os.path.exists( path_pkl ) is not True:        
            ## INITIALIZE HBOND OBJECT
            hbond_obj = Triplets( **self.__dict__ )
            
            ## CREATE PLACE HOLDER
            triplets = []
            
            ## PREVENT RAM OVERLOAD BY ITERATIVELY LOADING TRAJECTORY
            for traj in iterload_md_traj( path_traj    = self.sim_working_dir,
                                          input_prefix = self.input_prefix,
                                          iter_size    = self.iter_size ):
                ## PRINT PROGRESS
                if self.verbose is True:
                    print( "ANALYZING FRAMES {} TO {}".format( traj.time[0], traj.time[-1] ) )            
            
                ## CHECK TRAJ LENGTH
                if traj.n_frames > 1:
                    ## COMPUTE HBOND TRIPLETS
                    if self.n_procs > 1:
                        t = run_parallel( hbond_obj, traj, self.n_procs, 
                                          verbose = self.verbose )
                    else:
                        t = hbond_obj.compute( traj )
                else:
                    ## RUN IN SERIAL IF ONLY ONE FRAME
                    t = hbond_obj.compute( traj )

                ## APPEND RESULTS
                triplets += t
                
            ## SAVE TRIPLETS TO PKL
            save_pkl( triplets, path_pkl )
        else:
            ## LOAD FROM FILE
            triplets = load_pkl( path_pkl )
            
        ## RETURN RESULTS
        return triplets

## TRIPLETS CLASS OBJECT
class Triplets:
    """class object used to compute hydrogen bond triplets"""
    def __init__( self,
                  n_procs          = 1,
                  z_ref            = 0.0, 
                  z_cutoff         = 0.3, 
                  hbond_r_cutoff   = 0.35, 
                  hbond_phi_cutoff = 0.523598,
                  periodic         = True,
                  verbose          = True, 
                  print_freq       = 100,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.n_procs          = n_procs
        self.z_ref            = z_ref
        self.z_cutoff         = z_cutoff
        self.hbond_r_cutoff   = hbond_r_cutoff
        self.hbond_phi_cutoff = hbond_phi_cutoff
        self.periodic         = periodic
        self.verbose          = verbose
        self.print_freq       = print_freq
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)
                
    def __str__( self ):
        return "Hydrogen Bond Triplets"

    ## INSTANCE COMPUTING HBONDS FOR SINGLE FRAME
    def compute_single_frame( self, traj, frame = 0, ):
        """FUNCTION TO COMPUTE HBOND TRIPLETS FOR A SINGLE FRAME"""
        ## REDUCE TRAJ TO SINGLE FRAME
        traj = traj[frame]
        
        ## GET POTENTIAL DONOR AND ACCEPTOR INDICES
        atoms = [ atom for atom in traj.topology.atoms if atom.element.symbol in [ "N", "O" ] ]
        atom_indices = np.array([ atom.index for atom in atoms ])
        
        ## COMPUTE DISTANCE VECTORS BETWEEN REF POINT AND DONORS/ACCEPTORS
        ref_coords = [ 0.5*traj.unitcell_lengths[0,0], 0.5*traj.unitcell_lengths[0,1], self.z_ref ]
        z_dist = compute_displacements( traj,
                                        atom_indices = atom_indices,
                                        box_dimensions = traj.unitcell_lengths,
                                        ref_coords = ref_coords,
                                        periodic = self.periodic )[:,2]
        
        ## REDUCE ATOMS TO THOSE INSIDE CUTOFFS
        mask = np.logical_and( z_dist > -(self.z_cutoff + 0.5), # always includes head groups
                               z_dist < self.z_cutoff )
        mask_sliced = np.logical_and( z_dist > -(self.z_cutoff + 0.5), # always includes head groups
                                      z_dist < self.z_cutoff + 1.05*self.hbond_r_cutoff ) # include hbonders above
        
        ## MASK OUT TARGET ATOMS AND ATOMS TO SLICE
        target_atoms = atom_indices[mask]
        atoms_to_slice = atom_indices[mask_sliced]
        
        ## ADD HYDROGENS BACK TO ATOM LIST
        atom_indices_to_slice = []
        for aa in atoms_to_slice:
            group = [ aa ]
            for one, two in traj.topology.bonds:
                if aa == one.index and two.element.symbol == "H":
                    group.append( two.index )
                elif one.element.symbol == "H" and aa == two.index:
                    group.append( one.index )
            atom_indices_to_slice += group
        atom_indices_to_slice = np.array(atom_indices_to_slice)
        
        ## SLICE TRAJECTORY TO ONLY TARGET ATOMS AND THOSE WITHIN CUTOFF
        sliced_traj = traj.atom_slice( atom_indices_to_slice, inplace = False )
        
        ## COMPUTE TRIPLETS
        sliced_triplets = luzar_chandler( sliced_traj, 
                                          distance_cutoff = self.hbond_r_cutoff, 
                                          angle_cutoff    = self.hbond_phi_cutoff )

        ## GET ARRAY OF INDICES OF WHERE TRIPLETS CORRESPOND TO NDX_NEW
        triplets = np.array([ atom_indices_to_slice[ndx] for ndx in sliced_triplets.flatten() ])
        triplets = triplets.reshape( sliced_triplets.shape )   
        
        ## RETURN RESULTS
        return [ target_atoms, triplets ]
    
    ### FUNCTION TO COMPUTE FOR ALL FRAMES
    def compute( self, traj, frames = [] ):
        """FUNCTION TO COMPUTE THE HBOND TRIPLETS"""
        ## LOADING FRAMES TO TRAJECTORY
        if len(frames)>0:
            traj = traj[tuple(frames)]
            
        ## DEFINING TOTAL TRAJECTORY SIZE
        total_traj_size = traj.time.size

        ## TRACKING TIME
        timer = track_time()
        
        ## LOOPING THROUGH EACH TRAJECTORY FRAME
        for frame in np.arange(0, total_traj_size):
            if frame == 0:
                ## GETTING TRIPLET ANGLES FOR FIRST FRAME
                triplets = [self.compute_single_frame( traj = traj, frame = 0 )]
            else:
                ## COMPUTING TRIPLET ANGLES AND CONCATENATE
                triplets += [self.compute_single_frame( traj = traj, frame = frame )]
            ## PRINT PROGRESS IF VERBOSE
            if self.verbose is True and traj.time[frame] % self.print_freq == 0:
                 print( "  PID {}: Analyzing frame at {} ps".format( os.getpid(), traj.time[frame] ) )

        if self.verbose is True:
            ## OUTPUTTING TIME
            timer.time_elasped( "  PID " + str(os.getpid()) )
            
        ## RETURN RESULTS
        return triplets
    
## FUNCTION TO COMPUTE HBONDS USING LUZAR-CHANDLER CRITERIA
def luzar_chandler( traj, 
                    distance_cutoff = 0.35, 
                    angle_cutoff    = 0.523598 ):
    R'''Identify hydrogen bonds based on cutoffs for the Donor...Acceptor
    distance and H-Donor...Acceptor angle. Works best for a single trajectory
    frame, anything larger is prone to memory errors.
    
    The criterion employed is :math:'\\theta > \\pi/6' (30 degrees) and 
    :math:'r_\\text{Donor...Acceptor} < 3.5 A'.
    
    The donors considered by this method are NH and OH, and the acceptors 
    considered are O and N.
    
    Input
    -----
    traj : md.Trajectory
        An mdtraj trajectory.
    distance_cutoff : 0.35 nm (3.5 A)
        Default 'r_\\text{Donor..Acceptor}' distance
    angle_cutoff : '\\pi/6' (30 degrees)
        Default '\\theta' cutoff
    
    Output
    ------
    hbonds : np.array, shape=[n_hbonds, 3], dtype=int
        An array containing the indices atoms involved in each of the identified 
        hydrogen bonds. Each row contains three integer indices, '(d_i, h_i, a_i)',
        such that 'd_i' is the index of the donor atom , 'h_i' the index of the 
        hydrogen atom, and 'a_i' the index of the acceptor atom involved in a 
        hydrogen bond which occurs (according to the definition above).
        
    References
    ----------
    Luzar, A. & Chandler, D. Structure and hydrogen bond dynamics of water–
    dimethyl sulfoxide mixtures by computer simulations. J. Chem. Phys. 98, 
    8160–8173 (1993).
    '''    
    def _get_bond_triplets( traj ):    
        def get_donors(e0, e1):
            # Find all matching bonds
            elems = set((e0, e1))
            atoms = [ (one, two) for one, two in traj.topology.bonds 
                      if set((one.element.symbol, two.element.symbol)) == elems]
    
            # Get indices for the remaining atoms
            indices = []
            for a0, a1 in atoms:
                pair = (a0.index, a1.index)
                # make sure to get the pair in the right order, so that the index
                # for e0 comes before e1
                if a0.element.symbol == e1:
                    pair = pair[::-1]
                indices.append(pair)
    
            return indices
    
        nh_donors = get_donors('N', 'H')
        oh_donors = get_donors('O', 'H')
        xh_donors = np.array(nh_donors + oh_donors)
    
        if len(xh_donors) == 0:
            # if there are no hydrogens or protein in the trajectory, we get
            # no possible pairs and return nothing
            return np.zeros((0, 3), dtype=int)
    
        acceptor_elements = frozenset(('O', 'N'))
        acceptors = [ a.index for a in traj.topology.atoms 
                      if a.element.symbol in acceptor_elements ]
    
        # Make acceptors a 2-D numpy array
        acceptors = np.array(acceptors)[:, np.newaxis]
    
        # Generate the cartesian product of the donors and acceptors
        xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
        acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
        bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))
    
        # Filter out self-bonds
        self_bond_mask = (bond_triplets[:, 0] == bond_triplets[:, 2])
        return bond_triplets[np.logical_not(self_bond_mask), :]
    
    def _compute_bounded_geometry( traj, 
                                   triplets, 
                                   distance_indices = [ 0, 2 ], 
                                   angle_indices = [ 1, 0, 2 ] ):
            '''this function computes the distances between the atoms involved in
            the hydrogen bonds and the H-donor...acceptor angle using the law of 
            cosines.
            
            Inputs
            ------
            traj : md.traj
            triplets : np.array, shape[n_possible_hbonds, 3], dtype=int
                An array containing the indices of all possible hydrogen bonding triplets
            distance_indices : [LIST], [ donor_index, acceptor_index ], default = [ 0, 2 ]
                A list containing the position indices of the donor and acceptor atoms
            angle_indices : [LIST], [ h_index, donor_index, acceptor_index ], default = [ 1, 0, 2 ]
                A list containing the position indices of the H, donor, and acceptor 
                atoms. Default is H-donor...acceptor angle
              
            Outputs
            -------
            distances : np.array, shape[n_possible_hbonds, 1], dtype=float
                An array containing the distance between the donor and acceptor atoms
            angles : np.array, shape[n_possible_hbonds, 1], dtype=float
                An array containing the triplet angle between H-donor...acceptor atoms
            '''  
            # Calculate the requested distances
            distances = md.compute_distances( traj,
                                              triplets[ :, distance_indices ],
                                              periodic = True )
            
            # Calculate angles using the law of cosines
            abc_pairs = zip( angle_indices, angle_indices[1:] + angle_indices[:1] )
            abc_distances = []
            
            # calculate distances (if necessary)
            for abc_pair in abc_pairs:
                if set( abc_pair ) == set( distance_indices ):
                    abc_distances.append( distances )
                else:
                    abc_distances.append( md.compute_distances( traj, triplets[ :, abc_pair ], ) )
                    
            # Law of cosines calculation to find the H-Donor...Acceptor angle
            #            c**2 = a**2 + b**2 - 2*a*b*cos(C)
            #                        acceptor
            #                          /\
            #                         /  \
            #                      c /    \ b
            #                       /      \ 
            #                      /______(_\
            #                     H    a     donor
            a, b, c = abc_distances
            cosines = ( a ** 2 + b ** 2 - c ** 2 ) / ( 2 * a * b )
            np.clip(cosines, -1, 1, out=cosines) # avoid NaN error
            angles = np.arccos(cosines)
            
            return distances, angles

    if traj.topology is None:
        raise ValueError( 'hbond requires that traj contain topology information' )
    
    # get the possible donor-hydrogen...acceptor triplets    
    bond_triplets = _get_bond_triplets( traj )
    
    distances, angles = _compute_bounded_geometry( traj, bond_triplets )
    
    # Find triplets that meet the criteria
    presence = np.logical_and( distances < distance_cutoff, angles < angle_cutoff )
    
    return bond_triplets.compress( presence.flatten(), axis = 0 )

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

    ## INITIALIZE CLASS
    height_obj = SamHeight( traj                = traj,
                            sim_working_dir     = path_traj,
                            input_prefix        = input_prefix,
                            recompute_interface = False )
  
    ## WILLARD-CHANDLER INTERFACE
    wc_ref = height_obj.willard_chandler( per_frame = False )
    
    ## HBOND TRIPLETS
    triplets_obj = Triplets( sim_working_dir  = path_traj,
                             input_prefix     = input_prefix,
                             n_procs          = 1,
                             z_ref            = wc_ref, 
                             z_cutoff         = 0.3, 
                             hbond_r_cutoff   = 0.35, 
                             hbond_phi_cutoff = 0.523598,
                             periodic         = True,
                             verbose          = True, 
                             print_freq       = 1, )
    
    ## COMPUTE SINGLE FRAME
    triplets_single_frame = triplets_obj.compute_single_frame( traj, frame = 0 )
    
    ## RUN HBONDS OBJECT IN SERIAL
    hbonds = HydrogenBonds( sim_working_dir          = path_traj,
                            input_prefix             = input_prefix,
                            n_procs                  = 1,
                            iter_size                = 10,
                            z_ref                    = "willard_chandler", 
                            z_cutoff                 = 0.3, 
                            hbond_r_cutoff           = 0.35, 
                            hbond_phi_cutoff         = 0.523598,
                            periodic                 = True,
                            recompute_hbond_triplets = True,
                            verbose                  = True, 
                            print_freq               = 10, )
    ## COMPUTE IN SERIAL
    triplets_serial = hbonds.triplets()
    
    ## RUN HBONDS OBJECT IN PARALLEL
    hbonds = HydrogenBonds( sim_working_dir          = path_traj,
                            input_prefix             = input_prefix,
                            n_procs                  = 2,
                            iter_size                = 10,
                            z_ref                    = "willard_chandler", 
                            z_cutoff                 = 0.3, 
                            hbond_r_cutoff           = 0.35,
                            hbond_phi_cutoff         = 0.523598,
                            periodic                 = True,
                            recompute_hbond_triplets = True,
                            verbose                  = True, 
                            print_freq               = 10, )
    ## COMPUTE IN SERIAL
    triplets_parallel = hbonds.triplets()
    
    ## CHECK SERIAL AND PARALLEL RUNS
    if len(triplets_serial) == len(triplets_parallel):
        if np.all(triplets_serial[-1][0] == triplets_parallel[-1][0]):
            if np.all(triplets_serial[-1][1] == triplets_parallel[-1][1]):
                print( "\nSERIAL-PARALLEL MATCH: True" )
            else:
                print( "\nSERIAL-PARALLEL MATCH: False" )
        else:
            print( "\nSERIAL-PARALLEL MATCH: False" )

    ## PRINT OUT RESULTS
    print( "SINGLE FRAME TARGET ATOMS SIZE: {}".format( triplets_single_frame[0].shape ) )
    print( "SINGLE FRAME TRIPLETS SIZE: {}".format( triplets_single_frame[1].shape ) )
      
#    ## BENCHMARK ANALYSIS TOOL
#    NUM_EVAL_RUNS = 4
#
#    print('Evaluating Sequential Implementation...')
#    ANALYSIS_ATTR["n_procs"] = 1
#    sequential_result = compute_triplet_angle_distribution( **ANALYSIS_ATTR ) # "warm up", ensure cache is in consistent state
#    sequential_time = 0
#    for i in range(NUM_EVAL_RUNS):
#        start = time.perf_counter()
#        compute_triplet_angle_distribution( **ANALYSIS_ATTR )
#        sequential_time += time.perf_counter() - start
#    sequential_time /= NUM_EVAL_RUNS
#
#    print('Evaluating Parallel Implementation...')
#    n_procs = 4
#    ANALYSIS_ATTR["n_procs"] = n_procs
#    parallel_result = compute_triplet_angle_distribution( **ANALYSIS_ATTR )  # "warm up"
#    parallel_time = 0
#    for i in range(NUM_EVAL_RUNS):
#        start = time.perf_counter()
#        compute_triplet_angle_distribution( **ANALYSIS_ATTR )
#        parallel_time += time.perf_counter() - start
#    parallel_time /= NUM_EVAL_RUNS
#
#    if np.allclose( sequential_result, parallel_result ) is False:
#        raise Exception('sequential_result and parallel_result do not match.')
#    print('Average Sequential Time: {:.2f} ms'.format(sequential_time*1000))
#    print('Average Parallel Time: {:.2f} ms'.format(parallel_time*1000))
#    print('Speedup: {:.2f}'.format(sequential_time/parallel_time))
#    print('Efficiency: {:.2f}%'.format(100*(sequential_time/parallel_time)/n_procs))
