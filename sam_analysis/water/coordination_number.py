"""
coordination_number.py 
script contains functions to compute water molecule coordination number

CREATED ON: 02/02/2021

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
from sam_analysis.core.misc_tools import track_time

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## CLASS TO COMPUTE WATER COORDINATION NUMBER
class WaterCoordination:
    """class object used to compute water triplet angles"""
    def __init__( self,
                  sim_working_dir = None,
                  input_prefix    = None,
                  target_angle    = 48,
                  water_residues  = [ "SOL", "HOH" ],
                  recompute_coord = False,
                  verbose         = True, 
                  print_freq      = 100,
                  n_procs         = 1,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.sim_working_dir = sim_working_dir
        self.input_prefix    = input_prefix
        self.target          = target_angle
        self.water_residues  = water_residues
        self.recompute_coord = recompute_coord
        self.verbose         = verbose
        self.print_freq      = print_freq
        self.n_procs         = n_procs
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

    def __str__( self ):
        return "Water Coordination Number"

    ## COMPUTE DISTRIBUTIONS
    def distribution( self ):
        ## CREATE X AXIS
        self.x = np.arange( 0, 21, 1 )
        
        ## COMPUTE COORDINATION NUMBER
        coord, triplet5, triplet6 = self.number()
        
        ## COMPUTE TOTAL DISTRIBUTION
        y_all = np.histogram( coord[:,1], bins = 21, range = ( 0, 21 ) )[0]

        ## COMPUTE TARGET DISTRIBUTION
        mask = coord[:,5].astype("bool")
        target_coords = coord[mask,:]

        ## HISTOGRAM WITH COMPONENTS
        y_target    = np.zeros( shape = (21,) )
        y_target_ww = np.zeros( shape = (21,) )
        y_target_ss = np.zeros( shape = (21,) )
        for ii in range(len(target_coords[:,1])):
            ## CREATE INTS
            tc  = np.floor( target_coords[ii,1] ).astype("int")
            tcw = np.floor( target_coords[ii,2] ).astype("int")
            tcs = np.floor( target_coords[ii,3] ).astype("int")

            ## HISTOGRAMS
            y_target[tc]    += 1.
            y_target_ww[tc] += tcw / tc
            y_target_ss[tc] += tcs / tc

        # # TOTAL
        # y_target = np.histogram( target_coords[:,1], bins = 11, range = ( 0, 10 ) )[0]
        # # WATER-WATER
        # y_target_ww = np.histogram( target_coords[:,2], bins = 11, range = ( 0, 10 ) )[0]
        # # SAM-WATER
        # y_target_ss = np.histogram( target_coords[:,3], bins = 11, range = ( 0, 10 ) )[0]

        ## HISTOGRAM TRIPLETS
        self.theta    = np.arange( 0, 180, 2 )
        self.triplet5 = np.histogram( triplet5, bins = 90, range = ( 0, 180 ) )[0]
        self.triplet5 = self.triplet5 / np.max([ np.trapz( self.triplet5, dx = 2 ), 1 ])
        self.triplet6 = np.histogram( triplet6, bins = 90, range = ( 0, 180 ) )[0]
        self.triplet6 = self.triplet6 / np.max([ np.trapz( self.triplet6, dx = 2 ), 1 ])

        ## NORMALIZE
        # ALL
        all_constant  = np.max([ np.trapz( y_all, dx = 1 ), 1 ])
        self.dist_all = y_all / all_constant
        # self.dist_all = y_all
        # TARGET
        target_constant        = np.max([ np.trapz( y_target, dx = 1 ), 1 ])
        self.dist_target       = y_target / target_constant
        self.dist_target_water = y_target_ww / target_constant
        self.dist_target_sam   = y_target_ss / target_constant
        # self.dist_target       = y_target
        # self.dist_target_water = y_target_ww
        # self.dist_target_sam   = y_target_ss

    ## COORDINATATION NUMBER
    def number( self ):   
        ## PATH TO PKL
        out_name = self.input_prefix + "_coordination_number.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )

        ## LOAD DATA
        if self.recompute_coord is True or \
         os.path.exists( path_pkl ) is not True:
            ## INITIALIZE TRIPLET CLASS
            triplet_angle_obj = TripletAngleDistribution( **self.__dict__ )

            ## GET TRIPLETS
            triplets = triplet_angle_obj.angles
            
            ## LOAD TRAJ
            traj = load_md_traj( path_traj    = self.sim_working_dir,
                                 input_prefix = self.input_prefix )        
             ## GET ATOM TYPES
            atoms = [ atom for atom in traj.topology.atoms
                      if atom.element.symbol in [ "N", "O" ] ]
            water_atom_indices = np.array([ atom.index for atom in atoms 
                                            if atom.residue.name in self.water_residues ])

            ## INITIALIZE COORD NUM OBJECT
            coord_obj = CoordNum( triplets      = triplets,
                                  water_indices = water_atom_indices,
                                  **self.__dict__ )
                            
            ## COMPUTE DENSITY FIELDS
            coord = run_parallel( coord_obj, traj, self.n_procs,
                                  verbose = self.verbose, append = False )

            ## SAVE DICTIONARY TO PKL
            save_pkl( coord, path_pkl )
        else:
            ## LOAD FROM FILE
            coord = load_pkl( path_pkl )
        
        ## UNPACK RESULTS
        results  = coord[0][0]
        results5 = coord[0][1]
        results6 = coord[0][2]
        for cc in coord[1:]:
            results  += cc[0]
            results5 += cc[1]
            results6 += cc[2]
    
        ## RETURN RESULTS
        return np.array(results), np.array(results5), np.array(results6)

## COORDINATION NUMBER CLASS
class CoordNum:
    """class object used to compute coordination numbers"""
    def __init__( self,
                  n_procs       = 1,
                  triplets      = [],
                  target        = 48,
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
        return "Coordination Number"

    ## INSTANCE COMPUTING COORDINATION FOR A SINGLE FRAME
    def compute_single_frame( self, traj, triplets_all, target_triplets ):
        """FUNCTION TO COMPUTE COORDINATION FOR A SINGLE FRAME"""            
        ## LOOP THROUGH FRAMES
        coordination = []
        coord_5_triplets = []
        coord_6_triplets = []
        
        ## GET TRAJ FRAME
        frame = traj.time[0]
        
        ## TARGET PER FRAME
        target_frame_mask     = target_triplets[:,0] == frame
        target_triplets_frame = target_triplets[ target_frame_mask, : ]

        ## ALL TRIPLETS PER FRAME
        frame_mask     = triplets_all[:,0] == frame
        triplets_frame = triplets_all[ frame_mask, : ]

        ## UNIQUE WATERS
        target_waters_frame = np.unique(target_triplets_frame[:,2])
        waters_frame        = np.unique(triplets_frame[:,2])
        
        ## COORDINATION PER FRAME
        for ii in waters_frame:
            ## COUNT NEIGHBORS ALL WATERS
            tt  = triplets_frame[(triplets_frame[:,2] == ii),:]
            tt_neighbors = np.unique( np.hstack(( tt[:,3], tt[:,4] )) )
            num = len( tt_neighbors )

            ## WATER-WATER-WATER MASK
            www_mask = np.logical_and( np.isin( tt[:,3], self.water_indices ),
                                       np.isin( tt[:,4], self.water_indices ) )
            
            ## COMPUTE COORD TRIPLETS
            coord_triplets = list(tt[www_mask,1])

            ## DON'T ADD IF SIZE 0
            if len(coord_triplets) > 0:
                ## APPEND TRIPLET ANGLES WITH COORD = 5
                if num == 5:
                    coord_5_triplets += coord_triplets
                
                ## APPEND TRIPLET ANGLES WITH COORD = 6
                if num == 6:
                    coord_6_triplets += coord_triplets

            ## COUNT WATER-WATER NEIGHBORS
            ww_num = np.sum( np.isin( tt_neighbors, self.water_indices ) )

            ## COUNT SAM-WATER NEIGHBORS
            ss_num = num - ww_num

            ## MARK IF TARGET WATER
            is_target = 0
            if ii in target_waters_frame:
                is_target = 1
                
            ## UPDATE RESULTS
            # FRAME, COORD ALL, WATER-WATER COORD, SAM-WATER COORD, WATER ID, HAS TARGET ANGLE
            coordination.append( [ frame, num, ww_num, ss_num, ii, is_target ] )

        ## RETURN RESULTS
        return [ coordination, coord_5_triplets, coord_6_triplets ]
    
    ## FUNCTION TO COMPUTE FOR ALL FRAMES
    def compute( self, traj, ):
        ## GENERATE MASKS
        water_mask = np.isin( self.triplets[:,2], self.water_indices, invert = False )

        ## ALL WATER TRIPLETS
        triplets_all = self.triplets[water_mask,:]

        ## FIND TARGET INSTANCES
        target_instances = np.floor( triplets_all[:,1] / 2 ).astype("int") == self.target // 2

        ## TARGET TRIPLETS
        target_triplets = triplets_all[target_instances,:]

        ## TRACKING TIME
        timer = track_time()

        ## LOOPING THROUGH EACH TRAJECTORY FRAME
        for ii in range(0, traj.time.size):
            ## GETTING COORDINATION
            cc = self.compute_single_frame( traj            = traj[ii], 
                                            triplets_all    = triplets_all, 
                                            target_triplets = target_triplets )
            if ii == 0:
                ## UNPACK RESULTS
                coordination     = cc[0]
                coord_5_triplets = cc[1]
                coord_6_triplets = cc[2]
            else:
                ## UNPACK RESULTS AND APPEND
                coordination     += cc[0]
                coord_5_triplets += cc[1]
                coord_6_triplets += cc[2]
                
            ## PRINT PROGRESS IF VERBOSE
            if self.verbose is True and traj.time[ii] % self.print_freq == 0:
                 print( "  PID {}: Analyzing frame at {} ps".format( os.getpid(), traj.time[ii] ) )
                
        if self.verbose is True:
            ## OUTPUTTING TIME
            timer.time_elasped( "  PID " + str(os.getpid()) )

        ## RETURN RESULTS
        return [ coordination, coord_5_triplets, coord_6_triplets ]

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE WATER COORDINATION
    coord_obj = WaterCoordination( **kwargs )

    ## COMPUTE COORDINATION DIST
    coord_obj.distribution()
    
    ## RETURN RESULTS
    return { "coord"                   : coord_obj.x,
             "coord_dist"              : coord_obj.dist_all,
             "coord_target_dist"       : coord_obj.dist_target,
             "coord_target_dist_water" : coord_obj.dist_target_water,
             "coord_target_dist_sam"   : coord_obj.dist_target_sam,
             "triplet_theta"           : coord_obj.theta,
             "triplet_coord_5"         : coord_obj.triplet5,
             "triplet_coord_6"         : coord_obj.triplet6,  }

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## IMPORT CHECK SERVER PATH
    from sam_analysis.core.check_tools import check_server_path

    ## TESTING DIRECTORY
    test_dir = r"/home/bdallin/python_projects/sam_visualize/sam_visualize/sams"
    
    ## SAM DIRECTORY
    sam_dir = r"sam_ch3"

    ## TRAJ NAME
    traj_name   = r"sam_prod"
    
    ## WORKING DIR
    working_dir = os.path.join(test_dir, sam_dir)
    working_dir = check_server_path( working_dir )
        
    ## TARGET ANGLE
    target = 48

    ## INITIALIZE WATER COORDINATION
    coord_obj = WaterCoordination( sim_working_dir = working_dir,
                                   input_prefix    = traj_name,
                                   target_angle    = 48,
                                   recompute_coord = True, )

    ## COMPUTE COORDINATION DIST
    coord_obj.distribution()

    x  = coord_obj.x
    y  = coord_obj.dist_all
    yt = coord_obj.dist_target
    yw = coord_obj.dist_target_water
    ys = coord_obj.dist_target_sam
    
    # ## PRINT OUT RESULTS
    # print( "COORDINATION ARRAY SIZE: {}".format( coord.shape ) )
