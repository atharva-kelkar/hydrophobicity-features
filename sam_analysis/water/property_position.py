"""
property_position.py 
script contains functions to position of molecule with a certain property

CREATED ON: 04/06/2021

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
## IMPORT SERIAL AND PARALLEL FUNCTIONS
from sam_analysis.core.parallel import run_parallel
## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORTING TRACKING TIME AND TRAJ SPLIT
from sam_analysis.core.misc_tools import split_list, combine_objects
## IMPORT TRIPLET DISTRIBUTION CLASS
from sam_analysis.water.triplet_angle import TripletAngleDistribution
## IMPORT MISC TOOLS
from sam_analysis.core.misc_tools import track_time, compute_com, end_group_atom_indices

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## CLASS TO COMPUTE WATER COORDINATION NUMBER
class WaterPropertyPosition:
    """class object used to water property position"""
    def __init__( self,
                  sim_working_dir  = None,
                  input_prefix     = None,
                  property_value   = 90,
                  water_residues   = [ "SOL", "HOH" ],
                  recompute_coords = False,
                  verbose          = True, 
                  print_freq       = 100,
                  n_procs          = 1,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.sim_working_dir  = sim_working_dir
        self.input_prefix     = input_prefix
        self.target           = property_value
        self.water_residues   = water_residues
        self.recompute_coords = recompute_coords
        self.verbose          = verbose
        self.print_freq       = print_freq
        self.n_procs          = n_procs
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

    def __str__( self ):
        return "Water Coordination Number"

    ## COMPUTE DISTRIBUTIONS
    def distribution( self, yz = False ):
        ## COMPUTE PROPERTY POSITIONS
        coords, xyz_range = self.property_positions()

        ## 1D DISTRIBUTION
        if yz is False:
            ## X
            bin_width = 0.4
            x_range = ( xyz_range[0][0], xyz_range[0][1] )
            self.x = np.arange( x_range[0], x_range[1], bin_width )

            ## COMPUTE TOTAL DISTRIBUTION
            y, xs = np.histogram( coords[:,0], bins = len(self.x) )

            ## ADJUST PBC
            xs = xs[:-1]
            xs[ xs < x_range[0] ] += x_range[1]
            xs[ xs > x_range[1] ] -= x_range[1]
            indices = np.argsort( xs )
            y = y[indices]

            ## NORMALIZE
            # ALL
            constant  = np.max([ np.trapz( y, dx = bin_width ), 1 ])
            self.dist = y #/ constant

    ## PROPERTY POSITIONS
    def property_positions( self ):   
        ## PATH TO PKL
        out_name = self.input_prefix + "_theta90_positions.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )

        ## GET UNITCELL
        xrange = ( 0, 5.19120 )
        yrange = ( 0, 5.99428 )
        zrange = ( 0, 11.69890 )
        xyz_range = [ xrange, yrange, zrange ]

        ## LOAD DATA
        if self.recompute_coords is True or \
         os.path.exists( path_pkl ) is not True:
            ## INITIALIZE TRIPLET CLASS
            triplet_angle_obj = TripletAngleDistribution( **self.__dict__ )

            ## GET TRIPLETS
            triplets = triplet_angle_obj.angles

            ## LOAD TRAJ
            traj = load_md_traj( path_traj    = self.sim_working_dir,
                                 input_prefix = self.input_prefix )
            
            ## GET UNITCELL
            xrange = ( 0, traj.unitcell_lengths[0,0] )
            yrange = ( 0, traj.unitcell_lengths[0,1] )
            zrange = ( 0, traj.unitcell_lengths[0,2] )
            xyz_range = [ xrange, yrange, zrange ]

            ## GET ATOM TYPES
            atoms = [ atom for atom in traj.topology.atoms
                      if atom.element.symbol in [ "N", "O" ] ]
            water_atom_indices = np.array([ atom.index for atom in atoms 
                                            if atom.residue.name in self.water_residues ])

            ## INITIALIZE THETA90 OBJECT
            ninety_obj = Theta90( triplets      = triplets,
                                  water_indices = water_atom_indices,
                                  **self.__dict__ )
                            
            ## COMPUTE DENSITY FIELDS
            coords = ninety_obj.compute( traj )

            ## GET END GROUP ATOM INDICES
            end_atom_indices = end_group_atom_indices( traj, labels = False )
            
            ## COMPUTE COM POSITION OF END GROUP
            end_group_coms = np.zeros( shape = ( traj.n_frames, 
                                                 len(end_atom_indices), 
                                                 3 ) )
            for ii, indices in enumerate(end_atom_indices):
                end_group_coms[:,ii,:] = compute_com( traj, indices )
            
            ## END GROUP XY CENTER
            x_shift = end_group_coms[:,:,0].mean() - 0.5*traj.unitcell_lengths[0,0]
            y_shift = end_group_coms[:,:,1].mean() - 0.5*traj.unitcell_lengths[0,1]

            ## SHIFT COORDS
            coords[:,0] += x_shift
            coords[:,1] += y_shift

            ## SAVE DICTIONARY TO PKL
            save_pkl( coords, path_pkl )
        else:
            ## LOAD FROM FILE
            coords = load_pkl( path_pkl )
            
        ## RETURN RESULTS
        return coords, xyz_range

## COORDINATION NUMBER CLASS
class Theta90:
    """class object used to compute theta90 position"""
    def __init__( self,
                  n_procs       = 1,
                  triplets      = [],
                  target        = 90,
                  water_indices = [],
                  verbose       = True,
                  print_freq    = 100,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.n_procs       = n_procs
        self.triplets      = triplets
        self.target        = target
        self.water_indices = water_indices
        self.verbose       = verbose
        self.print_freq    = print_freq

        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

    def __str__( self ):
        return "Theta 90 positions"
    
    ## FUNCTION TO COMPUTE FOR ALL FRAMES
    def compute( self, traj, ):
        ## GENERATE MASKS
        water1_mask = np.isin( self.triplets[:,2], self.water_indices, invert = False )
        water2_mask = np.isin( self.triplets[:,3], self.water_indices, invert = False )
        water3_mask = np.isin( self.triplets[:,4], self.water_indices, invert = False )
        water_mask = water1_mask * water2_mask * water3_mask

        ## ALL WATER TRIPLETS
        triplets_all = self.triplets[water_mask,:]

        ## FIND TARGET INSTANCES
        target_instances = np.floor( triplets_all[:,1] / 2 ).astype("int") == self.target // 2

        ## TARGET TRIPLETS
        target_triplets = triplets_all[target_instances,:]

        ## TRACKING TIME
        timer = track_time()

        ## COORDS
        coords = np.empty( shape = ( 0, 3 ) )

        ## LOOPING THROUGH EACH TRAJECTORY FRAME
        for ii in range(traj.time.size):
            ## TARGET PER FRAME
            frame = traj.time[ii]
            target_frame_mask     = target_triplets[:,0] == frame
            target_triplets_frame = target_triplets[ target_frame_mask, : ].astype("int")

            ## TARGET POSITIONS
            # return target_triplets_frame[:,2]
            pos = traj.xyz[ii,target_triplets_frame[:,2],:]

            ## UPDATE
            coords = np.vstack(( coords, pos ))

            ## PRINT PROGRESS IF VERBOSE
            if self.verbose is True and traj.time[ii] % self.print_freq == 0:
                 print( "  PID {}: Analyzing frame at {} ps".format( os.getpid(), traj.time[ii] ) )

        if self.verbose is True:
            ## OUTPUTTING TIME
            timer.time_elasped( "  PID " + str(os.getpid()) )

        ## RETURN RESULTS
        return coords

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE POSITIONS
    obj = WaterPropertyPosition( **kwargs )

    ## COMPUTE DIST
    obj.distribution()
    
    ## RETURN RESULTS
    return { "theta90_x"    : obj.x,
             "theta90_dist" : obj.dist, }

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
    sam_dir    = r"sam_single_12x12_separated_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36"
    sample_dir = r"sample2"

    ## TRAJ NAME
    traj_name   = r"sam_prod"
    
    ## WORKING DIR
    working_dir = os.path.join(test_dir, sam_dir, sample_dir)
    working_dir = check_server_path( working_dir )
        
    ## TARGET ANGLE
    target = 90

    ## INITIALIZE WATER POSITION
    obj = WaterPropertyPosition( sim_working_dir  = working_dir,
                                 input_prefix     = traj_name,
                                 property_value   = target,
                                 recompute_coords = True )

    ## COMPUTE COORDINATION DIST
    obj.distribution()

    x  = obj.x
    y  = obj.dist
    
    # ## PRINT OUT RESULTS
    # print( "COORDINATION ARRAY SIZE: {}".format( coord.shape ) )
