"""
triplet_angle.py 
script contains functions to compute water molecule triplet angles in a MD trajectory

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
## IMPORT SERIAL AND PARALLEL FUNCTIONS
from sam_analysis.core.parallel import run_parallel
## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj, load_md_traj_frame, iterload_md_traj
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORT MISC TOOLS
from sam_analysis.core.misc_tools import track_time, compute_displacements

##############################################################################
# FUNCTIONS AND CLASSES
##############################################################################
## CLASS TO COMPUTE WATER TRIPLET ANGLE DISTRIBUTION
class TripletAngleDistribution:
    """class object used to compute water triplet angles"""
    def __init__( self,
                  sim_working_dir         = None,
                  input_prefix            = None,
                  z_ref                   = "willard_chandler",
                  z_cutoff                = 0.3,
                  theta_range             = ( 0, 180. ),
                  theta_bin_width         = 2.,
                  water_residues          = [ "SOL", "HOH" ],
                  n_procs                 = 1,
                  iter_size               = 1000,
                  periodic                = True,
                  recompute_triplet_angle = False,
                  verbose                 = True, 
                  print_freq              = 100,
                  split_traj              = True,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.sim_working_dir         = sim_working_dir
        self.input_prefix            = input_prefix
        self.z_ref                   = z_ref
        self.z_cutoff                = z_cutoff
        self.range                   = theta_range
        self.bin_width               = theta_bin_width
        self.water_residues          = water_residues
        self.n_procs                 = n_procs
        self.iter_size               = iter_size
        self.periodic                = periodic
        self.recompute_triplet_angle = recompute_triplet_angle
        self.verbose                 = verbose
        self.print_freq              = print_freq
        self.split_traj              = split_traj
        
        ## SET UP DISTRIBUTION
        self.n_bins = np.floor( theta_range[-1] / theta_bin_width ).astype("int")
        self.theta  = np.arange( theta_range[0] + 0.5*theta_bin_width, 
                                 theta_range[-1] + 0.5*theta_bin_width, 
                                 theta_bin_width )
        
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
                  
        ## GET ANGLES
        self.angles = self.triplets()        
        
        ## LOAD TRAJ (SINGLE FRAME)
        traj = load_md_traj_frame( path_traj    = self.sim_working_dir,
                                   input_prefix = self.input_prefix )        
        
        ## GET ATOM TYPES
        atoms = [ atom for atom in traj.topology.atoms
                  if atom.element.symbol in [ "N", "O" ] ]
        self.water_atom_indices = np.array([ atom.index for atom in atoms 
                                             if atom.residue.name in water_residues ])        
        ## GENERATE MASKS
        sam1_mask   = np.isin( self.angles[:,2], self.water_atom_indices, invert = True )
        sam2_mask   = np.isin( self.angles[:,3], self.water_atom_indices, invert = True )
        sam3_mask   = np.isin( self.angles[:,4], self.water_atom_indices, invert = True )
        water1_mask = np.isin( self.angles[:,2], self.water_atom_indices, invert = False )
        water2_mask = np.isin( self.angles[:,3], self.water_atom_indices, invert = False )
        water3_mask = np.isin( self.angles[:,4], self.water_atom_indices, invert = False )
        self.sam_sam_sam_mask       = sam1_mask * sam2_mask * sam3_mask
        self.sam_sam_water_mask     = sam1_mask * sam2_mask * water3_mask
        self.sam_water_sam_mask     = sam1_mask * water2_mask * sam3_mask
        self.sam_water_water_mask   = sam1_mask * water2_mask * water3_mask
        self.water_sam_sam_mask     = water1_mask * sam2_mask * sam3_mask
        self.water_sam_water_mask   = water1_mask * sam2_mask * water3_mask
        self.water_water_sam_mask   = water1_mask * water2_mask * sam3_mask
        self.water_water_water_mask = water1_mask * water2_mask * water3_mask
                        
    def __str__( self ):
        return "Water Triplet Angles"
    
    ## INSTANCE COMPUTING TRIPLET ANGLES
    def triplets( self ):
        ## PATH TO PKL
        out_name = self.input_prefix + "_triplet_angles.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )

        ## LOAD DATA
        if self.recompute_triplet_angle is True or \
           os.path.exists( path_pkl ) is not True:        
            ## INITIALIZE TRIPLET ANGLE OBJECT
            triplet_angle_obj = TripletAngle( **self.__dict__ )

            ## SPLIT TRAJ
            if self.split_traj is True:
                ## CREATE PLACE HOLDER
                triplet_angles = np.empty( shape = ( 0, 5 ) )
                ## PREVENT RAM OVERLOAD BY ITERATIVELY LOADING TRAJECTORY
                for traj in iterload_md_traj( path_traj    = self.sim_working_dir,
                                              input_prefix = self.input_prefix,
                                              iter_size    = self.iter_size ):
                    ## PRINT PROGRESS
                    if self.verbose is True:
                        print( "ANALYZING FRAMES {} TO {}".format( traj.time[0], traj.time[-1] ) )
                    
                    ## CHECK TRAJ LENGTH
                    if traj.n_frames > 1:
                        ## COMPUTE TRIPLET ANGLES
                        if self.n_procs > 1:
                            ta = run_parallel( triplet_angle_obj, traj, self.n_procs, 
                                            verbose = self.verbose )
                        else:
                            ta = triplet_angle_obj.compute( traj )
                    else:
                        ## RUN IN SERIAL IF ONLY ONE FRAME
                        ta = triplet_angle_obj.compute( traj )
                    
                    ## APPEND RESULTS
                    triplet_angles = np.vstack(( triplet_angles, ta ))
            else:
                ## LOAD TRAJ
                traj = load_md_traj( path_traj    = self.sim_working_dir,
                                     input_prefix = self.input_prefix )
                ## PRINT PROGRESS
                if self.verbose is True:
                    print( "ANALYZING FRAMES {} TO {}".format( traj.time[0], traj.time[-1] ) )
                
                ## CHECK TRAJ LENGTH
                if traj.n_frames > 1:
                    ## COMPUTE TRIPLET ANGLES
                    if self.n_procs > 1:
                        ta = run_parallel( triplet_angle_obj, traj, self.n_procs, 
                                        verbose = self.verbose )
                    else:
                        ta = triplet_angle_obj.compute( traj )
                else:
                    ## RUN IN SERIAL IF ONLY ONE FRAME
                    ta = triplet_angle_obj.compute( traj )

                ## STORE RESULTS
                triplet_angles = ta
            
            ## CONVERT TO NUMPY ARRAY
            triplet_angles = np.array( triplet_angles )
            
            ## SAVE TRIPLETSGRID TO PKL
            save_pkl( triplet_angles, path_pkl )
        else:
            ## LOAD FROM FILE
            triplet_angles = load_pkl( path_pkl )
            
        ## RETURN RESULTS
        return triplet_angles 
    
    ## INSTANCE COMPUTING TOTAL ANGLE DISTRIBUTION
    def dist_total( self ):
        """function to histogram total angle distributions"""        
        ## HISTOGRAM ANGLES
        dist = np.histogram( self.angles[:,1], bins = self.n_bins, range = self.range )[0]
        
        ## NORMALIZE HISTOGRAM
        norm_const = np.max([ np.trapz( dist, dx = self.bin_width ), 1 ])
        norm_dist = dist / norm_const
        
        ## RETURN RESULTS
        return norm_dist

    ## INSTANCE COMPUTING SAM-SAM-SAM ANGLE DISTRIBUTION
    def dist_sam_sam_sam( self ):
        """function to histogram sam-sam-sam angle distributions"""        
        ## HISTOGRAM ANGLES
        dist = np.histogram( self.angles[self.sam_sam_sam_mask,1], bins = self.n_bins, range = self.range )[0]
        
        ## NORMALIZE HISTOGRAM
        norm_const = np.max([ np.trapz( dist, dx = self.bin_width ), 1 ])
        norm_dist = dist / norm_const
        
        ## RETURN RESULTS
        return norm_dist

    ## INSTANCE COMPUTING SAM-SAM-WATER ANGLE DISTRIBUTION
    def dist_sam_sam_water( self ):
        """function to histogram sam-sam-water angle distributions"""        
        ## HISTOGRAM ANGLES
        dist = np.histogram( self.angles[self.sam_sam_water_mask,1], bins = self.n_bins, range = self.range )[0]
        
        ## NORMALIZE HISTOGRAM
        norm_const = np.max([ np.trapz( dist, dx = self.bin_width ), 1 ])
        norm_dist = dist / norm_const
        
        ## RETURN RESULTS
        return norm_dist

    ## INSTANCE COMPUTING SAM-WATER-SAM ANGLE DISTRIBUTION
    def dist_sam_water_sam( self ):
        """function to histogram sam-water-sam angle distributions"""        
        ## HISTOGRAM ANGLES
        dist = np.histogram( self.angles[self.sam_water_sam_mask,1], bins = self.n_bins, range = self.range )[0]
        
        ## NORMALIZE HISTOGRAM
        norm_const = np.max([ np.trapz( dist, dx = self.bin_width ), 1 ])
        norm_dist = dist / norm_const
        
        ## RETURN RESULTS
        return norm_dist

    ## INSTANCE COMPUTING SAM-WATER-WATER ANGLE DISTRIBUTION
    def dist_sam_water_water( self ):
        """function to histogram sam-water-water angle distributions"""        
        ## HISTOGRAM ANGLES
        dist = np.histogram( self.angles[self.sam_water_water_mask,1], bins = self.n_bins, range = self.range )[0]
        
        ## NORMALIZE HISTOGRAM
        norm_const = np.max([ np.trapz( dist, dx = self.bin_width ), 1 ])
        norm_dist = dist / norm_const
        
        ## RETURN RESULTS
        return norm_dist

    ## INSTANCE COMPUTING WATER-SAM-SAM ANGLE DISTRIBUTION
    def dist_water_sam_sam( self ):
        """function to histogram water-sam-sam angle distributions"""        
        ## HISTOGRAM ANGLES
        dist = np.histogram( self.angles[self.water_sam_sam_mask,1], bins = self.n_bins, range = self.range )[0]
        
        ## NORMALIZE HISTOGRAM
        norm_const = np.max([ np.trapz( dist, dx = self.bin_width ), 1 ])
        norm_dist = dist / norm_const
        
        ## RETURN RESULTS
        return norm_dist

    ## INSTANCE COMPUTING WATER-SAM-WATER ANGLE DISTRIBUTION
    def dist_water_sam_water( self ):
        """function to histogram water-sam-water angle distributions"""        
        ## HISTOGRAM ANGLES
        dist = np.histogram( self.angles[self.water_sam_water_mask,1], bins = self.n_bins, range = self.range )[0]
        
        ## NORMALIZE HISTOGRAM
        norm_const = np.max([ np.trapz( dist, dx = self.bin_width ), 1 ])
        norm_dist = dist / norm_const
        
        ## RETURN RESULTS
        return norm_dist

    ## INSTANCE COMPUTING WATER-WATER-SAM ANGLE DISTRIBUTION
    def dist_water_water_sam( self ):
        """function to histogram water-water-sam angle distributions"""        
        ## HISTOGRAM ANGLES
        dist = np.histogram( self.angles[self.water_water_sam_mask,1], bins = self.n_bins, range = self.range )[0]
        
        ## NORMALIZE HISTOGRAM
        norm_const = np.max([ np.trapz( dist, dx = self.bin_width ), 1 ])
        norm_dist = dist / norm_const
        
        ## RETURN RESULTS
        return norm_dist

    ## INSTANCE COMPUTING WATER-WATER-WATER ANGLE DISTRIBUTION
    def dist_water_water_water( self ):
        """function to histogram sam-water-water angle distributions"""        
        ## HISTOGRAM ANGLES
        dist = np.histogram( self.angles[self.water_water_water_mask,1], bins = self.n_bins, range = self.range )[0]
        
        ## NORMALIZE HISTOGRAM
        norm_const = np.max([ np.trapz( dist, dx = self.bin_width ), 1 ])
        norm_dist = dist / norm_const
        
        ## RETURN RESULTS
        return norm_dist
    
## CLASS COMPUTING TRIPLET ANGLES
class TripletAngle:
    """Class to compute water triplet angles"""
    ## INITIALIZING
    def __init__( self,
                  n_procs                 = 1,
                  z_ref                   = 0.0,
                  z_cutoff                = 0.3, # Shell ACS Nano used 0.5, INDUS uses 0.3 nm cavity
                  r_cutoff                = 0.33,
                  water_residues          = [ "SOL", "HOH" ],
                  periodic                = True,
                  recompute_triplet_angle = False,
                  verbose                 = True, 
                  print_freq              = 100,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.n_procs                 = n_procs
        self.z_ref                   = z_ref
        self.z_cutoff                = z_cutoff
        self.r_cutoff                = r_cutoff
        self.water_residues          = water_residues
        self.periodic                = periodic
        self.recompute_triplet_angle = recompute_triplet_angle
        self.verbose                 = verbose
        self.print_freq              = print_freq
                
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)        
        
    def __str__( self ):
        return "Water Triplet Angles"
    
    ## INSTANCE COMPUTING TRIPLET ANGLES FOR SINGLE FRAME
    def compute_single_frame( self, traj, frame = 0, ):
        """FUNCTION TO COMPUTE HBOND TRIPLETS FOR A SINGLE FRAME"""
        ## REDUCE TRAJ TO SINGLE FRAME
        traj = traj[frame]
        
        ## GET HBONDING ATOMS INDICES
        atoms        = [ atom for atom in traj.topology.atoms 
                         if atom.element.symbol in [ "N", "O" ] ]
        atom_indices = np.array([ atom.index for atom in atoms ])
        
        ## IF Z CUTOFF -1 (BULK SIMULATION) USE ALL ATOMS
        if self.z_cutoff > 0.:        
            ## COMPUTE DISTANCE VECTORS BETWEEN REF POINT AND DONORS/ACCEPTORS
            ref_coords = [ 0.5*traj.unitcell_lengths[0,0], 0.5*traj.unitcell_lengths[0,1], self.z_ref ]
            z_dist = compute_displacements( traj,
                                            atom_indices   = atom_indices,
                                            box_dimensions = traj.unitcell_lengths,
                                            ref_coords     = ref_coords,
                                            periodic       = self.periodic )[:,2]
        
            ## REDUCE ATOMS TO THOSE INSIDE CUTOFFS
            mask = np.logical_and( z_dist > -(self.z_cutoff + 0.5), # always includes head groups
                                z_dist < self.z_cutoff )
            mask_sliced = np.logical_and( z_dist > -(self.z_cutoff + 0.5), # always includes head groups
                                        z_dist < self.z_cutoff + 1.05*self.r_cutoff ) # include hbonders above
            
            ## MASK OUT TARGET ATOMS AND ATOMS TO SLICE
            target_atom_indices = atom_indices[mask]
            sliced_atom_indices = atom_indices[mask_sliced]
        else:
            ## MASK OUT TARGET ATOMS AND ATOMS TO SLICE
            target_atom_indices = atom_indices
            sliced_atom_indices = atom_indices
    
        ## COMPUTE ATOM-ATOM CARTESIAN DISTANCES FOR FIRST OXYGEN
        atom_pairs = np.array([ [ aa, target_atom_indices[0] ] 
                                  for aa in sliced_atom_indices 
                                  if aa != target_atom_indices[0] ])
        dist_vector = md.compute_displacements( traj, 
                                                atom_pairs = atom_pairs,
                                                periodic   = self.periodic )
        ## REDUCE EMPTY DIMENSION
        dist_vector = dist_vector.squeeze()
        
        ## COMPUTE TRIPLET ANGLES
        results = self.triplet_angles( dist_vector, sliced_atom_indices, ndx = 0 )
            
        ## REPEAT FOR THE OTHER OXYGENS
        for ii, atom_ndx in enumerate(target_atom_indices[1:]):
            atom_pairs = np.array([ [ aa, atom_ndx ] 
                                      for aa in sliced_atom_indices
                                      if aa != atom_ndx ])
            dist_vector = md.compute_displacements( traj, 
                                                    atom_pairs = atom_pairs,
                                                    periodic = self.periodic )
            ## REDUCE EMPTY DIMENSION
            dist_vector = dist_vector.squeeze()
            
            ## COMPUTE TRIPLET ANGLES
            results += self.triplet_angles( dist_vector, sliced_atom_indices, ndx = ii+1 )
                
        ## ADD FRAME TO RESULTS
        results = [ [ traj.time[0] ] + rr for rr in results ]
        
        ## RETURN RESULTS
        return results
    
    ## FUNCTION TO COMPUTE FOR ALL FRAMES
    def compute( self, traj, frames = [] ):
        """FUNCTION TO COMPUTE THE WATER TRIPLETS"""
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
                triplets = self.compute_single_frame( traj = traj, frame = 0 )
            else:
                ## COMPUTING TRIPLET ANGLES AND CONCATENATE
                triplets += self.compute_single_frame( traj = traj, frame = frame )
                
            ## PRINT PROGRESS IF VERBOSE
            if self.verbose is True and traj.time[frame] % self.print_freq == 0:
                 print( "  PID {}: Analyzing frame at {} ps".format( os.getpid(), traj.time[frame] ) )
                
        if self.verbose is True:
            ## OUTPUTTING TIME
            timer.time_elasped( "  PID " + str(os.getpid()) )
                
        ## RETURN RESULTS
        return triplets
    
    ### FUNCTION TO COMPUTE TRIPLET ANGLES
    def triplet_angles( self, vectors, indices, ndx = 0 ):
        """COMPUTES TRIPLET ANGLES"""
        ## PLACE HOLDER
        result = []
        
        ## COMPUTE VECTOR MAGNITUDES
        magnitudes = np.sqrt( np.sum( vectors**2., axis = 1 ) )
        
        ## COMPUTE NUMBER NEIGHBORS IN CUTOFF
        n_neighbors = np.sum( magnitudes < self.r_cutoff )
        
        ## SORT NEIGHBORS AND KEEP N NEAREST
        sorted_neighbors = magnitudes.argsort()[:n_neighbors]
        
        ## LOOP THROUGH NEIGHBORS
        for nn, ii in enumerate( sorted_neighbors[:-1] ):
            for jj in sorted_neighbors[nn+1:]:
                ## DOT PRODUCT TO GET COS(THETA)
                cos_theta = np.sum( vectors[ii,:] * vectors[jj,:] ) / magnitudes[ii] / magnitudes[jj]
                
                ## ADJUST FOR ROUNDING ERROR
                cos_theta = np.round( cos_theta, 4 )
                
                ## UPDATE ANGLES AND TRIPLETS
                theta = np.rad2deg( np.arccos( cos_theta ) )
                result.append([ theta, indices[ndx], indices[ii], indices[jj] ])
        
        ## RETURN RESULTS
        return result

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE CLASS
    triplet_angle_obj = TripletAngleDistribution( **kwargs )
        
    ## TOTAL TRIPLET ANGLE DISTRIBUTION
    total_dist = triplet_angle_obj.dist_total()
    
    ## WATER-WATER-WATER TRIPLET ANGLE DISTRIBUTION
    water_water_water_dist = triplet_angle_obj.dist_water_water_water()
    
    ## WATER-SAM-SAM TRIPLET ANGLE DISTRIBUTION
    water_sam_sam_dist = triplet_angle_obj.dist_water_sam_sam()
    
    ## WATER-WATER-WATER TRIPLET ANGLE DISTRIBUTION
    water_water_sam_dist = triplet_angle_obj.dist_water_water_sam()
    
    ## WATER-SAM-WATER TRIPLET ANGLE DISTRIBUTION
    water_sam_water_dist = triplet_angle_obj.dist_water_sam_water()
    
    ## SAM-WATER-SAM TRIPLET ANGLE DISTRIBUTION
    sam_water_sam_dist = triplet_angle_obj.dist_sam_water_sam()
    
    ## SAM-SAM-WATER TRIPLET ANGLE DISTRIBUTION
    sam_sam_water_dist = triplet_angle_obj.dist_sam_sam_water()
    
    ## SAM-WATER-WATER TRIPLET ANGLE DISTRIBUTION
    sam_water_water_dist = triplet_angle_obj.dist_sam_water_water()
    
    ## SAM-SAM-SAM TRIPLET ANGLE DISTRIBUTION
    sam_sam_sam_dist = triplet_angle_obj.dist_sam_sam_sam()
    
    ## RETURN RESULTS
    return { "triplet_angle_theta"            : triplet_angle_obj.theta,
             "total_distribution"             : total_dist,
             "water_water_water_distribution" : water_water_water_dist,
             "water_sam_sam_distribution"     : water_sam_sam_dist,
             "water_water_sam_distribution"   : water_water_sam_dist,
             "water_sam_water_distribution"   : water_sam_water_dist,
             "sam_water_sam_distribution"     : sam_water_sam_dist,
             "sam_sam_water_distribution"     : sam_sam_water_dist,
             "sam_water_water_distribution"   : sam_water_water_dist,
             "sam_sam_sam_distribution"       : sam_sam_sam_dist, }
        
#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## IMPORT CHECK SERVER PATH
    from sam_analysis.core.check_tools import check_server_path
    ## IMPORT TRAJECTORY FUNCTION
    from sam_analysis.core.trajectory import load_md_traj
    
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
    #height_obj = SamHeight( traj                = traj,
    #                        sim_working_dir     = path_traj,
    #                        input_prefix        = input_prefix,
    #                        recompute_interface = False )
  
    ## WILLARD-CHANDLER INTERFACE
    #wc_ref = height_obj.willard_chandler( per_frame = False )
                
    ## WATER TRIPLET ANGLES
    #triplet_obj = TripletAngle( n_procs                 = 1,
    #                            z_ref                   = wc_ref,
    #                            z_cutoff                = 0.3,
    #                            r_cutoff                = 0.33,
    #                            water_residues          = [ "SOL", "HOH" ],
    #                            periodic                = True,
    #                            recompute_triplet_angle = True,
    #                            verbose                 = True, 
    #                            print_freq              = 10, )
    
    ## COMPUTE SINGLE FRAME TRIPLET ANGLES
    #triplet_angles = triplet_obj.compute_single_frame( traj, frame = 0 )
    
    ## RUN TRIPLET ANGLE IN SERIAL
    #triplet_angle_obj = TripletAngleDistribution( sim_working_dir         = path_traj,
    #                                              input_prefix            = input_prefix,
    #                                              z_ref                   = "willard_chandler",
    #                                              z_cutoff                = 0.3,
    #                                              theta_range             = ( 0, 180. ),
    #                                              theta_bin_width         = 2.,
    #                                              water_residues          = [ "SOL", "HOH" ],
    #                                              n_procs                 = 1,
    #                                              iter_size               = 10,
    #                                              periodic                = True,
    #                                              recompute_triplet_angle = True,
    #                                              verbose                 = True, 
    #                                              print_freq              = 10 )
    #triplet_angles_serial = triplet_angle_obj.angles

    ## RUN TRIPLET ANGLE IN PARALLEL
    triplet_angle_obj = TripletAngleDistribution( sim_working_dir         = path_traj,
                                                  input_prefix            = input_prefix,
                                                  z_ref                   = "willard_chandler",
                                                  z_cutoff                = 0.3,
                                                  theta_range             = ( 0, 180. ),
                                                  theta_bin_width         = 2.,
                                                  water_residues          = [ "SOL", "HOH" ],
                                                  n_procs                 = 2,
                                                  iter_size               = 10,
                                                  periodic                = True,
                                                  recompute_triplet_angle = True,
                                                  verbose                 = True, 
                                                  print_freq              = 10 )
    triplet_angles_parallel = triplet_angle_obj.angles
        
    ## TOTAL TRIPLET ANGLE DISTRIBUTION
    #total_dist = triplet_angle_obj.dist_total()
    
    ## WATER-WATER-WATER TRIPLET ANGLE DISTRIBUTION
    water_water_water_dist = triplet_angle_obj.dist_water_water_water()
    
    ## WATER-SAM-SAM TRIPLET ANGLE DISTRIBUTION
    #water_sam_sam_dist = triplet_angle_obj.dist_water_sam_sam()
    
    ## WATER-WATER-WATER TRIPLET ANGLE DISTRIBUTION
    #water_water_sam_dist = triplet_angle_obj.dist_water_water_sam()
    
    ## WATER-SAM-WATER TRIPLET ANGLE DISTRIBUTION
    #water_sam_water_dist = triplet_angle_obj.dist_water_sam_water()
    
    ## SAM-WATER-SAM TRIPLET ANGLE DISTRIBUTION
    #sam_water_sam_dist = triplet_angle_obj.dist_sam_water_sam()
    
    ## SAM-SAM-WATER TRIPLET ANGLE DISTRIBUTION
    #sam_sam_water_dist = triplet_angle_obj.dist_sam_sam_water()
    
    ## SAM-WATER-WATER TRIPLET ANGLE DISTRIBUTION
    #sam_water_water_dist = triplet_angle_obj.dist_sam_water_water()
    
    ## SAM-SAM-SAM TRIPLET ANGLE DISTRIBUTION
    #sam_sam_sam_dist = triplet_angle_obj.dist_sam_sam_sam()
    
    ## CHECK SERIAL AND PARALLEL RUNS
    #print( "\nSERIAL-PARALLEL MATCH: {}".format( str(np.all( triplet_angles_serial == triplet_angles_parallel )) ) )
    
    ## PRINT OUT RESULTS
    #print( "SINGLE FRAME TRIPLET ANGLE SIZE: {}".format( len(triplet_angles) ) )
    #print( "TOTAL DISTRIBUTION SIZE: {}".format( total_dist.shape ) )
    #print( "WATER-WATER-WATER DISTRIBUTION SIZE: {}\n".format( water_water_water_dist.shape ) )
    #print( "WATER-SAM-SAM DISTRIBUTION SIZE: {}".format( water_sam_sam_dist.shape ) )
    #print( "WATER-WATER-SAM ONLY DISTRIBUTION SIZE: {}".format( water_water_sam_dist.shape ) )
    #print( "WATER-SAM-WATER DISTRIBUTION SIZE: {}".format( water_sam_water_dist.shape ) )
    #print( "SAM-WATER-SAM DISTRIBUTION SIZE: {}".format( sam_water_sam_dist.shape ) )
    #print( "SAM-SAM-WATER DISTRIBUTION SIZE: {}".format( sam_sam_water_dist.shape ) )
    #print( "SAM-WATER-WATER DISTRIBUTION SIZE: {}".format( sam_water_water_dist.shape ) )
    #print( "SAM-SAM-SAM DISTRIBUTION SIZE: {}".format( sam_sam_sam_dist.shape ) )
    
    ## RESULTS
    data = { "theta"   : triplet_angles_parallel,
             "p_theta" : water_water_water_dist }
    save_pkl( data, "triplet_angle_data.pkl" )
 
