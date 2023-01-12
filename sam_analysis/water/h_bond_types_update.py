"""
h_bond_types.py 
script contains functions to compute hbond types in a MD trajectory

CREATED ON: 11/24/2020

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
import numpy as np  # Used to do math functions

## IMPORT HEIGHT FUNCTION
from sam_analysis.sam.sam_height import SamHeight
## IMPORT SERIAL AND PARALLEL FUNCTIONS
from sam_analysis.core.parallel import run_parallel
## IMPORT HBOND CLASS
from sam_analysis.water.h_bond import HydrogenBonds
## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORTING TRACKING TIME AND TRAJ SPLIT
from sam_analysis.core.misc_tools import split_list, combine_objects

##############################################################################
# HBOND FUNCTIONS
##############################################################################
## HBOND TYPES CLASS
class HydrogenBondTypes:
    """class object used to compute hydrogen bond types"""
    def __init__( self,
                  triplet_list     = None,
                  sim_working_dir  = None,
                  input_prefix     = None,
                  n_procs          = 1,
                  z_ref            = "willard_chandler", 
                  z_cutoff         = 0.3, 
                  r_cutoff         = 0.35, 
                  angle_cutoff     = 0.523598,
                  hbond_types      = [ "sam-sam", 
                                       "sam-water", 
                                       "water-water" ],
                  water_residues   = [ "SOL", "HOH" ],
                  recompute_hbonds = False,
                  verbose          = True, 
                  print_freq       = 100,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.sim_working_dir  = sim_working_dir
        self.input_prefix     = input_prefix
        self.n_procs          = n_procs
        self.z_ref            = z_ref
        self.z_cutoff         = z_cutoff
        self.r_cutoff         = r_cutoff
        self.angle_cutoff     = angle_cutoff
        self.hbond_types      = hbond_types
        self.water_residues   = water_residues
        self.recompute_hbonds = recompute_hbonds
        self.verbose          = verbose
        self.print_freq       = print_freq      

        ## REMOVE TRAJ IF IN KWARGS (CAUSES MEMORY ISSUE IF PRELOADED)
        kwargs.pop( "traj", None )
                
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
                
        ## COMPUTE HBOND TRIPLETS
        hbonds = HydrogenBonds( **self.__dict__ )
        
        ## COMPUTE TRIPLETS
        self.triplet_list = hbonds.triplets()
        
    def __str__( self ):
        return "Hydrogen Bond Characterization"
    
    ## INSTANCE COMPUTING HBOND TYPES
    def types( self, traj ):
        ## PATH TO PKL
        out_name = self.input_prefix + "_hbond_types.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )
        
        ## LOAD DATA
        if self.recompute_hbonds is True or \
         os.path.exists( path_pkl ) is not True:
            ## INITIALIZE HBOND OBJECT
            types_obj = Types( **self.__dict__ )
            
            if self.n_procs > 1 \
               and traj.n_frames >= self.n_procs:
                ## CREATE PARALLIZABLE OBJECTS
                traj_chunks    = split_list( alist = traj, wanted_parts = self.n_procs )
                triplet_chunks = split_list( alist = self.triplet_list, 
                                             wanted_parts = self.n_procs )
                ## FREE UP MEMORY
                del traj
                
                ## COMBINE PARALLIZABLE OBJECT
                traj_triplets = combine_objects( traj_chunks, triplet_chunks )
                                
                ## COMPUTE DENSITY FIELDS
                types = run_parallel( types_obj, traj_triplets, self.n_procs,
                                      verbose = self.verbose, split_traj = False )

            else:
                ## RUN IN SERIAL
                types = types_obj.compute( [ traj, self.triplet_list ] )

            ## COMBINE RESULTS
            types_combined = types[0]
            ## APPEND LIST
            for ii in types[1:]:
                types_combined = np.vstack(( types_combined, ii ))
    
            ## SAVE TRIPLETSGRID TO PKL
            save_pkl( types_combined, path_pkl )
        else:
            ## LOAD FROM FILE
            types_combined = load_pkl( path_pkl )
        
        ## RETURN RESULTS
        return types_combined
    
    ## INSTANCE COMPUTING HBOND TYPES
    def sort_types( self, traj ):
        """function sorting hbond types"""
        ## PATH TO PKL
        out_name = self.input_prefix + "_hbond_sorted_types.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )
            
        ## GET NUMBER OF FRAMES
        n_frames = traj.n_frames
        
        ## LOAD DATA
        if self.recompute_hbonds is True or \
           os.path.exists( path_pkl ) is not True:                
            ## COMPUTE HBOND TYPES
            types = self.types( traj )
                
            ## INITIALIZE HBOND OBJECT
            sort_types_obj = SortedTypes( types = types, 
                                          **self.__dict__ )
            ## RUN ANALYSIS        
            if self.n_procs > 1 \
               and traj.n_frames >= self.n_procs:
                ## CREATE PARALLIZABLE OBJECTS
                traj_chunks    = split_list( alist = traj, wanted_parts = self.n_procs )
                triplet_chunks = split_list( alist = self.triplet_list, 
                                             wanted_parts = self.n_procs )                
                ## FREE UP MEMORY
                del traj
                
                ## COMBINE PARALLIZABLE OBJECT
                traj_triplets = combine_objects( traj_chunks, triplet_chunks )
                
                ## COMPUTE DENSITY FIELDS
                types = run_parallel( sort_types_obj, traj_triplets, self.n_procs,
                                      verbose = self.verbose, split_traj = False )
                
            else:
                ## RUN IN SERIAL
                types = sort_types_obj.compute( [ traj, self.triplet_list ] )
                
            ## COMBINE RESULTS
            types_combined = np.hstack(( types[0][0], types[0][1], types[0][2] ))
            
            ## APPEND LIST
            for ii in range(len(types[1:])):
                add = np.hstack(( types[ii][0], types[ii][1], types[ii][2] ))
                types_combined = np.vstack(( types_combined, add ))
                    
            ## COMBINE RESULT IN A MORE WORKABLE FORMAT
            n_types  = 3
            hbond_atom_indices = np.array(sorted(set( types_combined[:,1] )))
            
            ## CREATE PLACEHOLDER ARRAY
            hbond_array = np.empty(( n_frames, len(hbond_atom_indices), n_types ))
            hbond_array[:] = np.NaN
            
            ## FILL ARRAY FRAME BY FRAME        
            for ff in range(n_frames):
                ## GET TYPES IN FRAME
                sam_sam     = types[ff][0]
                sam_water   = types[ff][1]
                water_water = types[ff][2]
                
                ## TARGET ATOMS
                target_atoms = sam_sam[:,1]
                
                ## FILL ARRAY HBOND BY HBOND
                for ii, hh in enumerate(hbond_atom_indices):
                    ## GET TARGET INDEX
                    target = np.where( target_atoms == hh )[0]
                    
                    ## ONLY IF THERE IS A TARGET
                    if len(target) > 0:
                        ## EXTRACT INDEX
                        target_ndx = target[0]
                        
                        ## UPDATE HBOND ARRAY
                        hbond_array[ff,ii,0] = sam_sam[:,2][target_ndx]
                        hbond_array[ff,ii,1] = sam_water[:,2][target_ndx]
                        hbond_array[ff,ii,2] = water_water[:,2][target_ndx]
            
            ## SORTED TYPES
            sorted_types = { "indices"  : hbond_atom_indices,
                             "n_hbonds" : hbond_array }
            
            ## SAVE TRIPLETSGRID TO PKL
            save_pkl( sorted_types, path_pkl )
        else:
            ## LOAD FROM FILE
            sorted_types = load_pkl( path_pkl )
        
        ## RETURN RESULTS
        return sorted_types
            
## TRIPLETS CLASS OBJECT
class Types:
    """class object used to compute hydrogen bond types"""
    def __init__( self,
                  n_procs        = 1,
                  hbond_types    = [ "sam-sam", 
                                     "sam-water", 
                                     "water-water" ],
                  water_residues = [ "SOL", "HOH" ],
                  verbose        = True, 
                  print_freq     = 100,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.n_procs        = n_procs
        self.hbond_types    = hbond_types
        self.water_residues = water_residues
        self.verbose        = verbose
        self.print_freq     = print_freq

        ## HBOND TYPE KEY: 0 sam-sam, 1 sam-water, 1 water-sam, 2 water-water, 3 water-water outside cutoff
        self.hbond_key = { "sam-sam"     : 0,
                           "sam-water"   : 1,
                           "water-sam"   : 1,
                           "water-water" : 2 }
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

    def __str__( self ):
        return "Hydrogen Bond Types"

    ## INSTANCE COMPUTING HBONDS TYPES FOR SINGLE FRAME
    def compute_single_frame( self, traj, triplets_list, frame = 0, ):
        """FUNCTION TO COMPUTE HBOND TYPES FOR A SINGLE FRAME"""
        ## EXTRACT TRAJ AND TRIPLETS
        traj     = traj[frame]
#        target_atoms = triplets_list[frame][0]
        triplets = triplets_list[frame][1]
        
        ## SET UP PLACEHOLDER ARRAY ( frame, hbond_type, donor, acceptor )
        hbond = [] 
        
        ## LOOP THROUGH TRIPLETS TO SEPARATE INTO HBOND TYPES
        for ii, (donor, acceptor) in enumerate( triplets[:,[0,2]] ):
            ## ASSIGN DONOR TYPE
            if traj.topology.atom(donor).residue.name \
               not in self.water_residues:
                donor_type = "sam"
            else:
                donor_type = "water"
                
            ## ASSIGN ACCEPTOR TYPE
            if traj.topology.atom(acceptor).residue.name \
               not in self.water_residues:
                acceptor_type = "sam"
            else:
                acceptor_type = "water"
                
            ## COMBINE STRING
            hb_type = donor_type + '-' + acceptor_type
            
            ## APPEND TO DICTIONARY
            hbond.append( ( traj.time[0], self.hbond_key[hb_type], donor, acceptor ) )
            
        ## RETURN RESULTS
        return np.array(hbond)
    
    ### FUNCTION TO COMPUTE FOR ALL FRAMES
    def compute( self, traj_triplets, frames = [] ):
        """FUNCTION TO COMPUTE THE HBOND TYPES"""
        ## EXTRACT TRAJ AND TRIPLETS
        traj     = traj_triplets[0]
        triplets = traj_triplets[1]
        
        ## LOADING FRAMES TO TRAJECTORY
        if len(frames)>0:
            traj = traj[tuple(frames)]
            
        ## DEFINING TOTAL TRAJECTORY SIZE
        total_traj_size = traj.time.size
        
        ## LOOPING THROUGH EACH TRAJECTORY FRAME
        for frame in np.arange(0, total_traj_size):
            if frame == 0:
                ## GETTING TRIPLET ANGLES FOR FIRST FRAME
                types = [self.compute_single_frame( traj          = traj,
                                                    triplets_list = triplets,
                                                    frame         = 0 )]
            else:
                ## COMPUTING TRIPLET ANGLES AND CONCATENATE
                types += [self.compute_single_frame( traj          = traj,
                                                     triplets_list = triplets,
                                                     frame         = frame )]
            ## PRINT PROGRESS IF VERBOSE
            if self.verbose is True and traj.time[frame] % self.print_freq == 0:
                 print( "  PID {}: Analyzing frame at {} ps".format( os.getpid(), traj.time[frame] ) )
            
        ## RETURN RESULTS
        return types    

## TRIPLETS CLASS OBJECT
class SortedTypes:
    """class object used to compute hydrogen bond types"""
    def __init__( self,
                  types           = None,
                  n_procs         = 1,
                  verbose         = True, 
                  print_freq      = 100,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.types           = types
        self.n_procs         = n_procs
        self.verbose         = verbose
        self.print_freq      = print_freq
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

    def __str__( self ):
        return "Hydrogen Bond Sorted Types"

    ## INSTANCE COMPUTING HBONDS TYPES FOR SINGLE FRAME
    def compute_single_frame( self, traj, triplets_list, frame = 0, ):
        """FUNCTION TO COMPUTE HBOND TYPES FOR A SINGLE FRAME"""
        ## EXTRACT TRAJ AND TRIPLETS
        traj         = traj[frame]
        target_atoms = triplets_list[frame][0]
        
        ## REDUCE TYPES TO FRAME
        frame_mask = self.types[:,0] == traj.time[0]
        types      = self.types[frame_mask,:]
        
        ## CREATE PLACEHOLDERS
        sam_sam_hbonds     = []
        sam_water_hbonds   = []
        water_water_hbonds = []
        
        ## LOOP THROUGH TARGET ATOMS COUNTING HBONDS
        for ta in target_atoms:                
            ## CREATE TYPE MASKS
            sam_sam_mask     = types[:,1] == 0
            sam_water_mask   = types[:,1] == 1
            water_water_mask = types[:,1] == 2
            
            ## CREATE TARGET ATOM MASKS
            ta_donor_mask    = types[:,2] == ta
            ta_acceptor_mask = types[:,3] == ta
            ## CREATE UNION
            ta_mask = np.logical_or( ta_donor_mask, ta_acceptor_mask )
            
            ## FILTER OUT HBONDS
            sam_sam     = types[np.logical_and( ta_mask, sam_sam_mask ),:]
            sam_water   = types[np.logical_and( ta_mask, sam_water_mask ),:]
            water_water = types[np.logical_and( ta_mask, water_water_mask ),:]
        
            ## COUNT HBONDS PER FRAME
            sam_sam_hbonds.append( ( traj.time[0], ta, sam_sam.shape[0] ) )
            sam_water_hbonds.append( ( traj.time[0], ta, sam_water.shape[0] ) )
            water_water_hbonds.append( ( traj.time[0], ta, water_water.shape[0] ) )
        
        ## RETURN RESULTS
        return [ np.array(sam_sam_hbonds),
                 np.array(sam_water_hbonds),
                 np.array(water_water_hbonds) ]
    
    ### FUNCTION TO COMPUTE FOR ALL FRAMES
    def compute( self, traj_triplets, frames = [] ):
        """FUNCTION TO COMPUTE THE HBOND SORTED TYPES"""
        ## EXTRACT TRAJ AND TRIPLETS
        traj     = traj_triplets[0]
        triplets = traj_triplets[1]
        
        ## LOADING FRAMES TO TRAJECTORY
        if len(frames)>0:
            traj = traj[tuple(frames)]
            
        ## DEFINING TOTAL TRAJECTORY SIZE
        total_traj_size = traj.time.size
        
        ## LOOPING THROUGH EACH TRAJECTORY FRAME
        for frame in np.arange(0, total_traj_size):
            if frame == 0:
                ## GETTING TRIPLET ANGLES FOR FIRST FRAME
                types = [self.compute_single_frame( traj          = traj,
                                                    triplets_list = triplets,
                                                    frame         = 0 )]
            else:
                ## COMPUTING TRIPLET ANGLES AND CONCATENATE
                types += [self.compute_single_frame( traj          = traj,
                                                     triplets_list = triplets,
                                                     frame         = frame )]
            ## PRINT PROGRESS IF VERBOSE
            if self.verbose is True and traj.time[frame] % self.print_freq == 0:
                 print( "  PID {}: Analyzing frame at {} ps".format( os.getpid(), traj.time[frame] ) )
            
        ## RETURN RESULTS
        return types  

## NUMBER OF HBONDS CLASS
class NumberHydrogenBonds:
    """class object used to compute number of hydrogen bond for various types"""
    def __init__( self,
                  normal_end_group = True,
                  sim_working_dir  = None,
                  input_prefix     = None,
                  n_procs          = 1,
                  hbond_types      = [ "sam-sam", 
                                       "sam-water", 
                                       "water-water" ],
                  water_residues   = [ "SOL", "HOH" ],
                  recompute_hbonds = False,
                  verbose          = True, 
                  print_freq       = 100,
                  **kwargs ):
        ## REMOVE TRAJ IF IN KWARGS (CAUSES MEMORY ISSUE IF PRELOADED)
        kwargs.pop( "traj", None )

        ## LOAD TRAJ
        traj = load_md_traj( path_traj    = sim_working_dir,
                             input_prefix = input_prefix )
        
        ## RUN HBONDS OBJECT
        hbonds = HydrogenBondTypes( sim_working_dir  = sim_working_dir,
                                    input_prefix     = input_prefix,
                                    n_procs          = n_procs,
                                    hbond_types      = hbond_types,
                                    water_residues   = water_residues,
                                    recompute_hbonds = recompute_hbonds,
                                    verbose          = verbose, 
                                    print_freq       = print_freq,
                                    **kwargs )
        ## COMPUTE SORTED TYPES
        sorted_types = hbonds.sort_types( traj )
        
        ## SPLIT TO TYPES AND INDICES
        self.types   = sorted_types["n_hbonds"]
        self.indices = sorted_types["indices"]
        
        ## GET ATOM TYPES
        atoms = [ atom for atom in traj.topology.atoms
                  if atom.element.symbol in [ "N", "O" ] ]
        self.water_atom_indices = np.array([ atom.index for atom in atoms 
                                             if atom.residue.name in water_residues ])
        
        ## GENERATE MASKS
        self.sam_mask   = np.isin( self.indices, self.water_atom_indices, invert = True )
        self.water_mask = np.isin( self.indices, self.water_atom_indices, invert = False )
        
        ## CALCULATE NORMALIZING CONSTANTS
        self.n_sam    = self.sam_mask.sum()
        self.n_water  = self.water_mask.sum()
        self.n_frames = traj.n_frames
        
        ## ALTERNATIVE N_SAM (NORMALIZE BY END GROUP)
        if normal_end_group is True:
            non_dod = [ ligand for ligand in traj.topology.residues 
                        if ligand.name not in [ "SOL", "HOH", "DOD" ] ]
            ## GET NUMBER OF HBONDING END GROUPS
            self.n_sam = len(non_dod)
            
        ## NUMBER HBONDS ARRAY FOR DISTRIBUTIONS
        self.n_hbonds = np.arange( 0, 10, 1 )
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

    def __str__( self ):
        return "Number of Hydrogen Bonds"

    ## INSTANCE COMPUTING TOTAL NUMBER HBONDS
    def number_total( self, per_frame = False ):
        """computes average total number hbonds"""
        ## CREATE MASK
        mask = np.logical_or( self.sam_mask, self.water_mask )
        
        ## MASK HBONDS
        hbonds = self.types[:,mask,:]
        
        ## GET WATER NORMALIZING CONSTANTS (THIS IS DIFFERENT FOR EACH FRAME)
        water_hbonds = self.types[:,self.water_mask,0]
        n_water = np.nansum( ~np.isnan(water_hbonds), axis = 1 )
        
        ## SUM TYPES TOGETHER
        sum_types = np.nansum( hbonds, axis = 2 )
        
        ## SUM HBONDS
        sum_hbonds = np.nansum( sum_types, axis = 1 )
        
        ## NORMALIZE BY NUM. ATOMS
        norm_hbonds = sum_hbonds / ( self.n_sam + n_water )
        
        ## PER FRAME
        if per_frame is True:
            ## RETURN RESULTS
            return norm_hbonds
                    
        ## RETURN RESULTS
        return np.mean( norm_hbonds )
    
    ## INSTANCE COMPUTING NUMBER SAM-SAM HBONDS
    def number_sam_sam( self, per_frame = False ):
        """computes average number sam-sam hbonds"""
        ## MASK HBONDS
        hbonds = self.types[:,self.sam_mask,0]
        
        ## PER FRAME
        if per_frame is True:
            ## SUM HBONDS
            sum_hbonds = np.nansum( hbonds, axis = 1 )
            
            ## RETURN RESULTS
            return sum_hbonds / np.max([ self.n_sam, 1 ])
        
        ## SUM HBONDS
        sum_hbonds = np.nansum( hbonds )
        
        ## RETURN RESULTS
        return sum_hbonds / np.max([ self.n_sam, 1 ]) / self.n_frames

    ## INSTANCE COMPUTING TOTAL NUMBER HBONDS
    def number_sam_water( self, per_frame = False ):
        """computes average total number hbonds"""
        ## CREATE MASK
        mask = np.logical_or( self.sam_mask, self.water_mask )
        
        ## MASK HBONDS
        hbonds = self.types[:,mask,1]
        
        ## GET WATER NORMALIZING CONSTANTS (THIS IS DIFFERENT FOR EACH FRAME)
        water_hbonds = self.types[:,self.water_mask,1]
        n_water = np.nansum( ~np.isnan(water_hbonds), axis = 1 )
                
        ## SUM HBONDS
        sum_hbonds = np.nansum( hbonds, axis = 1 )
        
        ## NORMALIZE BY NUM. ATOMS
        norm_hbonds = sum_hbonds / ( self.n_sam + n_water )
        
        ## PER FRAME
        if per_frame is True:
            ## RETURN RESULTS
            return norm_hbonds
                    
        ## RETURN RESULTS
        return np.mean( norm_hbonds )
    
    ## INSTANCE COMPUTING NUMBER SAM-WATER HBONDS PER SAM
    def number_sam_water_per_sam( self, per_frame = False ):
        """computes average number sam-water hbonds per sam"""
        ## MASK HBONDS
        hbonds = self.types[:,self.sam_mask,1]
        
        ## PER FRAME
        if per_frame is True:
            ## SUM HBONDS
            sum_hbonds = np.nansum( hbonds, axis = 1 )
            
            ## RETURN RESULTS
            return sum_hbonds / np.max([ self.n_sam, 1 ])
        
        ## SUM HBONDS
        sum_hbonds = np.nansum( hbonds )
        
        ## RETURN RESULTS
        return sum_hbonds / np.max([ self.n_sam, 1 ]) / self.n_frames
    
    ## INSTANCE COMPUTING NUMBER SAM-WATER HBONDS PER WATER
    def number_sam_water_per_water( self, per_frame = False ):
        """computes average number sam-water hbonds per water"""
        ## MASK HBONDS
        hbonds = self.types[:,self.water_mask,1]
        
        ## PER FRAME
        if per_frame is True:            
            ## RETURN RESULTS
            return np.nanmean( hbonds, axis = 1 )
                
        ## RETURN RESULTS
        return np.mean( np.nanmean( hbonds, axis = 1 ) )
    
    ## INSTANCE COMPUTING NUMBER SAM-WATER HBONDS PER WATER
    def number_water_water( self, per_frame = False ):
        """computes average number water-water hbonds"""
        ## MASK HBONDS
        hbonds = self.types[:,self.water_mask,2]
        
        ## PER FRAME
        if per_frame is True:
            ## RETURN RESULTS
            return np.nanmean( hbonds, axis = 1 )
                
        ## RETURN RESULTS
        return np.mean( np.nanmean( hbonds, axis = 1 ) )

    ## INSTANCE COMPUTING TOTAL NUMBER HBONDS DISTRIBUTION
    def dist_total( self ):
        """computes number distribution of total hbonds"""
        ## CREATE MASK
        mask = np.logical_or( self.sam_mask, self.water_mask )
        
        ## MASK HBONDS
        hbonds = self.types[:,mask,:]
        
        ## NOT NAN MASK
        not_nan = ~np.isnan( hbonds )
        
        ## HISTOGRAM HBONDS
        dist = np.histogram( hbonds[not_nan], bins = 10, range = ( 0, 10 ) )[0]
        
        ## RETURN RESULTS
        norm_const = np.max([ np.trapz( dist, dx = 1 ), 1 ])
        return dist / norm_const

    ## INSTANCE COMPUTING NUMBER SAM-SAM HBONDS DISTRIBUTION
    def dist_sam_sam( self ):
        """computes number distribution of sam-sam hbonds"""
        ## MASK HBONDS
        hbonds = self.types[:,self.sam_mask,0]
        
        ## NOT NAN MASK
        not_nan = ~np.isnan( hbonds )
        
        ## HISTOGRAM HBONDS
        dist = np.histogram( hbonds[not_nan], bins = 10, range = ( 0, 10 ) )[0]
        
        ## RETURN RESULTS
        norm_const = np.max([ np.trapz( dist, dx = 1 ), 1 ])
        return dist / norm_const

    ## INSTANCE COMPUTING NUMBER SAM-WATER HBONDS DISTRIBUTION
    def dist_sam_water( self ):
        """computes number distribution of sam-water hbonds"""
        ## CREATE MASK
        mask = np.logical_or( self.sam_mask, self.water_mask )
        
        ## MASK HBONDS
        hbonds = self.types[:,mask,1]
        
        ## NOT NAN MASK
        not_nan = ~np.isnan( hbonds )
        
        ## HISTOGRAM HBONDS
        dist = np.histogram( hbonds[not_nan], bins = 10, range = ( 0, 10 ) )[0]
        
        ## RETURN RESULTS
        norm_const = np.max([ np.trapz( dist, dx = 1 ), 1 ])
        return dist / norm_const

    ## INSTANCE COMPUTING NUMBER SAM-WATER HBONDS PER SAM DISTRIBUTION
    def dist_sam_water_per_sam( self ):
        """computes number distribution of sam-water hbonds per sam"""
        ## MASK HBONDS
        hbonds = self.types[:,self.sam_mask,1]
        
        ## NOT NAN MASK
        not_nan = ~np.isnan( hbonds )
        
        ## HISTOGRAM HBONDS
        dist = np.histogram( hbonds[not_nan], bins = 10, range = ( 0, 10 ) )[0]
        
        ## RETURN RESULTS
        norm_const = np.max([ np.trapz( dist, dx = 1 ), 1 ])
        return dist / norm_const
    
    ## INSTANCE COMPUTING NUMBER SAM-WATER HBONDS PER WATER DISTRIBUTION
    def dist_sam_water_per_water( self ):
        """computes number distribution of sam-water hbondsper water """
        ## MASK HBONDS
        hbonds = self.types[:,self.water_mask,1]
        
        ## NOT NAN MASK
        not_nan = ~np.isnan( hbonds )
        
        ## HISTOGRAM HBONDS
        dist = np.histogram( hbonds[not_nan], bins = 10, range = ( 0, 10 ) )[0]
        
        ## RETURN RESULTS
        norm_const = np.max([ np.trapz( dist, dx = 1 ), 1 ])
        return dist / norm_const
    
    ## INSTANCE COMPUTING NUMBER WATER-WATER HBONDS DISTRIBUTION
    def dist_water_water( self ):
        """computes number distribution of water-water hbonds"""
        ## MASK HBONDS
        hbonds = self.types[:,self.water_mask,2]
        
        ## NOT NAN MASK
        not_nan = ~np.isnan( hbonds )
        
        ## HISTOGRAM HBONDS
        dist = np.histogram( hbonds[not_nan], bins = 10, range = ( 0, 10 ) )[0]
        
        ## RETURN RESULTS
        norm_const = np.max([ np.trapz( dist, dx = 1 ), 1 ])
        return dist / norm_const

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE CLASS
    num_hbonds = NumberHydrogenBonds( **kwargs )
    
    ## COMPUTE TOTAL HBONDS
    num_total = num_hbonds.number_total( per_frame = False )
    
    ## COMPUTE SAM-SAM HBONDS
    num_sam_sam = num_hbonds.number_sam_sam( per_frame = False )
    
    ## COMPUTE SAM-WATER HBONDS
    num_sam_water = num_hbonds.number_sam_water( per_frame = False )
    
    ## COMPUTE SAM-WATER HBONDS PER SAM
    num_sam_water_per_sam = num_hbonds.number_sam_water_per_sam( per_frame = False )
    
    ## COMPUTE SAM-WATER HBONDS PER WATER
    num_sam_water_per_water = num_hbonds.number_sam_water_per_water( per_frame = False )
    
    ## COMPUTE WATER-WATER HBONDS PER WATER
    num_water_water = num_hbonds.number_water_water( per_frame = False )

    ## COMPUTE TOTAL HBONDS PER FRAME
    num_total_per_frame = num_hbonds.number_total( per_frame = True )
    
    ## COMPUTE SAM-SAM HBONDS PER FRAME
    num_sam_sam_per_frame = num_hbonds.number_sam_sam( per_frame = True )
    
    ## COMPUTE SAM-WATER HBONDS PER FRAME
    num_sam_water_per_frame = num_hbonds.number_sam_water( per_frame = True )
    
    ## COMPUTE SAM-WATER HBONDS PER SAM PER FRAME
    num_sam_water_per_sam_per_frame = num_hbonds.number_sam_water_per_sam( per_frame = True )
    
    ## COMPUTE SAM-WATER HBONDS PER WATER PER FRAME
    num_sam_water_per_water_per_frame = num_hbonds.number_sam_water_per_water( per_frame = True )
    
    ## COMPUTE WATER-WATER HBONDS PER WATER PER FRAME
    num_water_water_per_frame = num_hbonds.number_water_water( per_frame = True )
    
    ## COMPUTE N HBONDS
    dist_n = num_hbonds.n_hbonds
    
    ## COMPUTE TOTAL HBOND DISTRIBUTION
    dist_total = num_hbonds.dist_total()
    
    ## COMPUTE SAM-SAM HBOND DISTRIBUTION
    dist_sam_sam = num_hbonds.dist_sam_sam()
    
    ## COMPUTE SAM-WATER HBOND DISTRIBUTION
    dist_sam_water = num_hbonds.dist_sam_water()
    
    ## COMPUTE SAM-WATER HBOND DISTRIBUTION PER SAM
    dist_sam_water_per_sam = num_hbonds.dist_sam_water_per_sam()
    
    ## COMPUTE SAM-WATER HBOND DISTRIBUTION PER WATER
    dist_sam_water_per_water = num_hbonds.dist_sam_water_per_water()
    
    ## COMPUTE WATER-WATER HBOND DISTRIBUTION
    dist_water_water = num_hbonds.dist_water_water()

    ## RETURN RESULTS
    return { "hbonds_total"                         : num_total,
             "hbonds_sam_sam"                       : num_sam_sam,
             "hbonds_sam_water"                     : num_sam_water,
             "hbonds_sam_water_per_sam"             : num_sam_water_per_sam,
             "hbonds_sam_water_per_water"           : num_sam_water_per_water,
             "hbonds_water_water"                   : num_water_water,
             "hbonds_total_per_frame"               : num_total_per_frame,
             "hbonds_sam_sam_per_frame"             : num_sam_sam_per_frame,
             "hbonds_sam_water_per_frame"           : num_sam_water_per_frame,
             "hbonds_sam_water_per_sam_per_frame"   : num_sam_water_per_sam_per_frame,
             "hbonds_sam_water_per_water_per_frame" : num_sam_water_per_water_per_frame,
             "hbonds_water_water_per_frame"         : num_water_water_per_frame,
             "hbonds_dist_n"                        : dist_n,
             "hbonds_dist_total"                    : dist_total,
             "hbonds_dist_sam_sam"                  : dist_sam_sam,
             "hbonds_dist_sam_water"                : dist_sam_water,
             "hbonds_dist_sam_water_per_sam"        : dist_sam_water_per_sam,
             "hbonds_dist_sam_water_per_water"      : dist_sam_water_per_water,
             "hbonds_dist_water_water"              : dist_water_water }
    
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
    #traj = load_md_traj( path_traj    = path_traj,
    #                     input_prefix = input_prefix )
    
    ## COMPUTE HBOND TRIPLETS
    #hbonds = HydrogenBonds( traj                     = traj,
    #                        sim_working_dir          = path_traj,
    #                        input_prefix             = input_prefix,
    #                        n_procs                  = 4,
    #                        z_ref                    = "willard_chandler", 
    #                        z_cutoff                 = 0.3, 
    #                        r_cutoff                 = 0.35, 
    #                        angle_cutoff             = 0.523598,
    #                        periodic                 = True,
    #                        recompute_hbond_triplets = False,
    #                        verbose                  = True, 
    #                        print_freq               = 10, )
    ## COMPUTE TRIPLETS
    #triplets = hbonds.triplets()
            
    ## HBOND TYPES SINGLE FRAME
    #hbond_types_obj = Types( n_procs         = 1,
    #                         hbond_types     = [ "sam-sam", 
    #                                             "sam-water", 
    #                                             "water-water" ],
    #                         water_residues  = [ "SOL", "HOH" ],
    #                         verbose         = True, 
    #                         print_freq      = 1 )
    
    ## COMPUTE SINGLE FRAME
    #types_single_frame = hbond_types_obj.compute_single_frame( traj, triplets, frame = 0 )

    ## SORT TYPES SINGLE FRAME
    #types_obj = SortedTypes( types           = types_single_frame,
    #                         n_procs         = 1,
    #                         verbose         = True, 
    #                         print_freq      = 10, )
    
    ## COMPUTE SINGLE FRAME
    #sorted_types_single_frame = types_obj.compute_single_frame( traj, triplets )
        
    ## RUN HBONDS OBJECT IN SERIAL
    #hbonds = HydrogenBondTypes( sim_working_dir          = path_traj,
    #                            input_prefix             = input_prefix,
    #                            n_procs                  = 1,
    #                            iter_size                = 10,
    #                            hbond_types              = [ "sam-sam", 
    #                                                         "sam-water", 
    #                                                         "water-water" ],
    #                            water_residues           = [ "SOL", "HOH" ],
    #                            recompute_hbonds         = True,
    #                            recompute_hbond_triplets = False,
    #                            verbose                  = True, 
    #                            print_freq               = 10, )
    ## COMPUTE TYPES IN SERIAL
    #types_serial = hbonds.types()
        
    ## COMPUTE SORTED TYPES IN SERIAL
    #sorted_types_serial = hbonds.sort_types()

    ## RUN HBONDS OBJECT IN PARALLEL
    hbonds = HydrogenBondTypes( sim_working_dir          = path_traj,
                                input_prefix             = input_prefix,
                                n_procs                  = 8,
                                iter_size                = 10,
                                hbond_types              = [ "sam-sam", 
                                                             "sam-water", 
                                                             "water-water" ],
                                water_residues           = [ "SOL", "HOH" ],
                                recompute_hbonds         = True,
                                recompute_hbond_triplets = True,
                                verbose                  = True, 
                                print_freq               = 100, )
    ## COMPUTE IN PARALLEL
    types_parallel = hbonds.types()
        
    ## COMPUTE SORTED TYPES IN PARALLEL
    sorted_types_parallel = hbonds.sort_types()   
    
    ## NUMBER HBONDS
    num_hbonds = NumberHydrogenBonds( normal_end_groups        = True,
                                      sim_working_dir          = path_traj,
                                      input_prefix             = input_prefix,
                                      n_procs                  = 1,
                                      iter_size                = 10,
                                      hbond_types              = [ "sam-sam", 
                                                                   "sam-water", 
                                                                   "water-water" ],
                                      water_residues           = [ "SOL", "HOH" ],
                                      recompute_hbonds         = True,
                                      recompute_hbond_triplets = False,
                                      verbose                  = True, 
                                      print_freq               = 10, )
    
    ## COMPUTE TOTAL HBONDS
    num_total = num_hbonds.number_total( per_frame = False )
    
    ## COMPUTE SAM-SAM HBONDS
    #num_sam_sam = num_hbonds.number_sam_sam( per_frame = False )
    
    ## COMPUTE SAM-WATER HBONDS
    num_sam_water = num_hbonds.number_sam_water( per_frame = False )
    
    ## COMPUTE SAM-WATER HBONDS PER SAM
    #num_sam_water_per_sam = num_hbonds.number_sam_water_per_sam( per_frame = False )
    
    ## COMPUTE SAM-WATER HBONDS PER WATER
    #num_sam_water_per_water = num_hbonds.number_sam_water_per_water( per_frame = False )
    
    ## COMPUTE WATER-WATER HBONDS PER WATER
    #num_water_water = num_hbonds.number_water_water( per_frame = False )

    ## COMPUTE TOTAL HBONDS PER FRAME
    #num_total_per_frame = num_hbonds.number_total( per_frame = True )
    
    ## COMPUTE SAM-SAM HBONDS PER FRAME
    #num_sam_sam_per_frame = num_hbonds.number_sam_sam( per_frame = True )
    
    ## COMPUTE SAM-WATER HBONDS PER FRAME
    #num_sam_water_per_frame = num_hbonds.number_sam_water( per_frame = True )
    
    ## COMPUTE SAM-WATER HBONDS PER SAM PER FRAME
    #num_sam_water_per_sam_per_frame = num_hbonds.number_sam_water_per_sam( per_frame = True )
    
    ## COMPUTE SAM-WATER HBONDS PER WATER PER FRAME
    #num_sam_water_per_water_per_frame = num_hbonds.number_sam_water_per_water( per_frame = True )
    
    ## COMPUTE WATER-WATER HBONDS PER WATER PER FRAME
    #num_water_water_per_frame = num_hbonds.number_water_water( per_frame = True )
    
    ## COMPUTE TOTAL HBOND DISTRIBUTION
    #dist_total = num_hbonds.dist_total()
    
    ## COMPUTE SAM-SAM HBOND DISTRIBUTION
    #dist_sam_sam = num_hbonds.dist_sam_sam()
    
    ## COMPUTE SAM-WATER HBOND DISTRIBUTION
    dist_sam_water = num_hbonds.dist_sam_water()
    
    ## COMPUTE SAM-WATER HBOND DISTRIBUTION PER SAM
    #dist_sam_water_per_sam = num_hbonds.dist_sam_water_per_sam()
    
    ## COMPUTE SAM-WATER HBOND DISTRIBUTION PER WATER
    #dist_sam_water_per_water = num_hbonds.dist_sam_water_per_water()
    
    ## COMPUTE WATER-WATER HBOND DISTRIBUTION
    #dist_water_water = num_hbonds.dist_water_water()
    
    ## CHECK SINGLE FRAME RUN
    #print( "SINGLE FRAME HBOND TYPES SIZE: {}".format( types_single_frame.shape ) )
    
    ## CHECK SINGLE FRAME RUN
    #print( "SINGLE FRAME SORTED HBOND TYPES SIZE: {}".format( sorted_types_single_frame[0].shape ) )
        
    ## CHECK SERIAL AND PARALLEL RUNS
    #print( "SERIAL-PARALLEL MATCH: {}".format( str(np.all( types_serial == types_parallel )) ) )
    
    ## CHECK SORTED SERIAL AND PARALLEL RUNS
    #print( "SORTED SERIAL-PARALLEL MATCH: {}".format( str(np.nansum(sorted_types_serial["n_hbonds"]) == np.nansum(sorted_types_parallel["n_hbonds"]) ) ) )

    ## PRINT OUT RESULTS
    #print( "TOTAL HBONDS: {}".format( num_total ) )
    #print( "SAM-SAM HBONDS: {}".format( num_sam_sam ) )
    #print( "SAM-WATER HBONDS: {}".format( num_sam_water ) )
    #print( "SAM-WATER PER SAM HBONDS: {}".format( num_sam_water_per_sam ) )
    #print( "SAM-WATER PER WATER HBONDS: {}".format( num_sam_water_per_water ) )
    #print( "WATER-WATER HBONDS: {}".format( num_water_water ) )
    #print( "TOTAL HBONDS PER FRAME SIZE: {}".format( num_total_per_frame.shape ) )
    #print( "SAM-SAM HBONDS PER FRAME SIZE: {}".format( num_sam_sam_per_frame.shape ) )
    #print( "SAM-WATER HBONDS PER FRAME SIZE: {}".format( num_sam_water_per_frame.shape ) )
    #print( "SAM-WATER PER SAM HBONDS PER FRAME SIZE: {}".format( num_sam_water_per_sam_per_frame.shape ) )
    #print( "SAM-WATER PER WATER HBONDS PER FRAME SIZE: {}".format( num_sam_water_per_water_per_frame.shape ) )
    #print( "WATER-WATER HBONDS PER FRAME SIZE: {}".format( num_water_water_per_frame.shape ) )
    #print( "TOTAL HBOND SIZE: {}".format( dist_total.shape ) )
    #print( "SAM-SAM HBOND SIZE: {}".format( dist_sam_sam.shape ) )
    #print( "SAM-WATER HBOND SIZE: {}".format( dist_sam_water.shape ) )
    #print( "SAM-WATER HBOND PER SAM SIZE: {}".format( dist_sam_water_per_sam.shape ) )
    #print( "SAM-WATER HBOND PER WATER SIZE: {}".format( dist_sam_water_per_water.shape ) )
    #print( "WATER-WATER HBOND SIZE: {}".format( dist_water_water.shape ) )

    data = { "hbond_total" : num_total,
             "hbond_sam_water" : num_sam_water,
             "hbond_dist_N" : num_hbonds.n_hbonds,
             "hbond_dist_sam_water" : dist_sam_water  }
    save_pkl( data, "hbond_data.pkl" )
