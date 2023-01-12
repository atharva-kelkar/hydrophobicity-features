"""
voronoi.py 
script contains various methods to characterize Voronoi cells of a SAM

CREATED ON: 09/28/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **
12/02/2020: cleaned up code and add more detailed comments

TODO:
"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT OS
import os
## IMPORT NUMPY
import numpy as np
## IMPORT VORONOI OBJECTS
from scipy.spatial import Voronoi, voronoi_plot_2d
## IMPORT PYPLOT
import matplotlib.pyplot as plt

## IMPORT GLOBAL LIGAND AND SOLVENT DATA
from sam_analysis.globals.ligands import LIGANDS
## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORT MISC TOOLS
from sam_analysis.core.misc_tools import compute_com, end_group_atom_indices
## IMPORT PLOTTING
from sam_analysis.plotting.jacs_single import JACS
from sam_analysis.plotting.plot_styles import UnbiasedDefaults

##############################################################################
# Voronoi tesselation class
##############################################################################
class SamVoronoi:
    R'''Computes the Voronoi tessellation of a 2D or 3D system using qhull. This
    uses scipy.spatial.Voronoi, accounting for pbc.
    
    qhull does not support pbc the box is expanded to include the periodic images.
    '''
    ## INITIALIZE CLASS
    def __init__( self,
                  traj              = None,
                  sim_working_dir   = None,
                  input_prefix      = None,
                  dimensions        = 2,
                  periodic          = True,
                  recompute_voronoi = False,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.traj              = traj
        self.sim_working_dir   = sim_working_dir
        self.input_prefix      = input_prefix
        self.dimensions        = dimensions
        self.periodic          = periodic
        self.recompute_voronoi = recompute_voronoi
        
        ## LOAD TRAJ
        if self.traj is None:
            self.traj = load_md_traj( path_traj    = sim_working_dir,
                                      input_prefix = input_prefix )
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)
            
    ## TESSELLATION METHOD COMPUTES VORONOI OBJECT
    def tessellation( self, per_frame = False ):
        ## PATH TO PKL
        if per_frame is not True:
            out_name = self.input_prefix + "_voronoi_tessellation.pkl"
        else:
            out_name = self.input_prefix + "_voronoi_tessellation_per_frame.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )
        
        ## LOAD DATA
        if self.recompute_voronoi is True or \
         os.path.exists( path_pkl ) is not True:
            ## LOAD TRAJ IN NOT INPUT
            if self.traj is None:
                self.traj = load_md_traj( path_traj    = self.sim_working_dir,
                                          input_prefix = self.input_prefix )
                
            ## GET END GROUP ATOM INDICES
            ligand_atom_indices, ligand_labels = end_group_atom_indices( self.traj, labels = True )
            
            ## COMPUTE COM POSITION OF END GROUP
            end_group_coms = np.zeros( shape = ( self.traj.n_frames, 
                                                 len(ligand_atom_indices), 
                                                 3 ) )
            for ii, indices in enumerate(ligand_atom_indices):
                end_group_coms[:,ii,:] = compute_com( self.traj, indices )
            
            ## IF PERIODIC REPLICATE IMAGES
            if self.periodic is True:
                end_group_coms = replicate_images( end_group_coms[...,:2], self.traj.unitcell_lengths[:,:2] )
                ligand_labels = ligand_labels * ( 3**self.dimensions )
            
            if per_frame is True:            
                ## COMPUTE VORONOI TESSELATION FRAME BY FRAME
                vor = []
                for ii in range(end_group_coms.shape[0]):
                    vor.append( Voronoi( end_group_coms[ii,...] ) )
                    
            else:
                ## IF NOT PER FRAME, COMPUTE TIME AVERAGE
                end_group_coms = end_group_coms.mean( axis = 0 )
                
                ## COMPUTE VORONOI TESSELATION (NOTE: Voronoi can only handle single frame, or surface avg)
                vor = Voronoi( end_group_coms )
            
            ## COMBINE RESULTS
            voron = [ vor, ligand_labels ]
            
            ## SAVE GRID TO PKL
            save_pkl( voron, path_pkl )
            
#            ## WRITE PDB FILE
#            if path_pdb is not None:
#                create_pdb( grid, path_pdb, traj )
        else:
             ## LOAD PKL
            voron = load_pkl( path_pkl )  
            
        ## RETURN VORONOI OBJ
        return voron[0], voron[1]
    
    ## INSTANCE TO COMPUTE POLYGON AREA
    def area( self, per_frame = False ):
        """Function to compute area of polygons"""
        ## COMPUTE VORONOI TESSELLATION
        vor, labels = self.tessellation( per_frame = per_frame )
        
        if per_frame is True:
            ## CREATE PLACE HOLDER
            areas = []
            ## LOOP THROUGH VORONOI OBJECTS FRAME BY FRAME
            for vv in vor:
                areas.append( self.shoelace( vv ) )
                
        else:
            ## COMPUTE AREA USING SHOELACE METHOD
            areas = self.shoelace( vor )
        
        ## CONVERT TO NUMPY ARRAY
        areas = np.array( areas )
        
        ## RETURN RESULTS
        return areas

    ## INSTANCE TO COMPUTE SAM COMPOSITION
    def polar_composition( self ):
        """Function to compute SAM composition"""
        ## RUN TESSELLATION
        vor, labels = self.tessellation( per_frame = False )
        
        ## SELECT POINTS AND CELLS ONLY IN THE SIMULATION BOX
        in_x   = np.logical_and( vor.points[:,0] > 0., 
                                 vor.points[:,0] < self.traj.unitcell_lengths[0,0] )
        in_y   = np.logical_and( vor.points[:,1] > 0, 
                                 vor.points[:,1] < self.traj.unitcell_lengths[0,1] )
        in_box = np.logical_and( in_x, in_y )
        
        ## REMOVE LIGANDS NOT IN BOX
        labels = [ label for label, ii in zip( labels, in_box ) if ii == 1 ]
        
        ## COUNT NONPOLAR GROUPS
        n_nonpolar = labels.count( "DOD" )
        
        ## CALCULATE POLAR COMPOSITION (ASSUMES BINARY MIXTURE)
        total = len(labels)
        composition = 1. - n_nonpolar / float(total)
        
        ## RETURN RESULTS
        return composition
            
    ## INSTANCE TO COMPUTE SHOELACE METHOD
    def shoelace( self, vor ):
        """Function computes area of a voronoi polygon using shoelace method"""
        ## SELECT POINTS AND CELLS ONLY IN THE SIMULATION BOX
        in_x   = np.logical_and( vor.points[:,0] > 0., 
                                 vor.points[:,0] < self.traj.unitcell_lengths[0,0] )
        in_y   = np.logical_and( vor.points[:,1] > 0, 
                                 vor.points[:,1] < self.traj.unitcell_lengths[0,1] )
        in_box = np.logical_and( in_x, in_y )
        
        ## REMOVE POLYGONS NOT IN BOX
        region_ndx = vor.point_region[in_box]
        
        ## CALCULATE THE AREA OF EACH CELL USING THE SHOELACE METHOD
        areas = []
        for ndx in region_ndx:
            region = vor.regions[ndx]
            x = vor.vertices[region,0]
            y = vor.vertices[region,1]
            shift_up = np.arange( -len(x)+1, 1 )
            shift_down = np.arange( -1, len(x)-1 )
            areas.append( np.abs( np.sum( 0.5 * x * ( y[shift_up] - y[shift_down] ) ) ) )
        
        ## RETURN RESULTS
        return areas
    
    ## DIAGRAM METHOD PLOTS VORONOI TESSELLATION
    def plot_diagram( self, vor, labels ):
        ## PATH TO FIGURE
        fig_path = os.path.join( self.sim_working_dir, 
                                 "output_files", 
                                 self.input_prefix + "_voronoi_diagram" )
        
        ## INITIALIZE PLOT DETAILS
        plot_details = JACS()
        plt.rcParams['axes.grid'] = False
        
        ## INITIALIZE PLOTTING AXES
        plot_axes = UnbiasedDefaults( "voronoi_diagram" )

        ## CREATE SUBPLOTS
        fig, ax = plt.subplots()
        ## ADJUST SUBPLOTS
        fig.subplots_adjust( left = 0.10, bottom = 0.10, right = 0.95, top = 0.95 )  
        
        ## PLOT A VORONOI DIAGRAM
        voronoi_plot_2d( vor, ax = ax, show_points = False, show_vertices = False,
                         line_colors = 'k', line_width = 2, line_alpha = 1.0 )
        
        ## PLOT POINTS
        for ii, label in enumerate( labels ):
            x = vor.points[ii,0]
            y = vor.points[ii,1]
            ax.plot( x, y, linestyle = "None", marker = "o", markersize = 8,
                     color = plot_details.colors[LIGANDS[label]] )

        ## SET X TICKS
        ax.set_xticks( plot_axes.major_xticks, minor = False ) # sets major ticks
        ax.set_xticks( plot_axes.minor_xticks, minor = True )  # sets minor ticks
        ax.set_xlim( 0.0, 5.0 ) # set bounds
        # SET X LABEL
        ax.set_xlabel( plot_axes.xlabel )

        ## SET Y TICKS
        ax.set_yticks( plot_axes.major_yticks, minor = False ) # sets major ticks
        ax.set_yticks( plot_axes.minor_yticks, minor = True )  # sets minor ticks
        ax.set_ylim( 0.0, 5.8 ) # set bounds
        # SET Y LABEL
        ax.set_ylabel( plot_axes.ylabel )

        ## RESIZE FIGURE TO DESIRED SIZE
        width = 5
        height = 5.5
        fig.set_size_inches( width, height ) # 5.5:5 aspect ratio
        if fig_path is not None:
            png = fig_path + ".png"
            svg = fig_path + ".svg"
            print( "FIGURE SAVED TO: {}".format( png ) )
            fig.savefig( png, dpi = 300, facecolor = 'w', edgecolor = 'w' )
            fig.savefig( svg, dpi = 300, facecolor = 'w', edgecolor = 'w' )

## COMPUTE THE BUFFER PARTICLES
def replicate_images( positions, box ):
    R'''Adds nearest image particles to box
    
    INPUT:
        positions: [np.array; size = (n_frames, n_atoms, 2 or 3)] array containing atom positions
        box: [np.array; size = (n_frames, 2 or 3)] simution box vectors
        
    OUTPUT:
        new_positions: [ np.array; size = ( n_frames, n_atoms*(9 or 27), 2 or 3)] array containing positions of real and image atoms
    '''
    ## DETERMINING DIMENSIONS OF DIAGRAM
    if positions.shape[2] < 3:
        is_2d = True
    else:
        is_2d = False
    
    ## UPDATE POSITIONS TO ACCOUNT FOR ALL SIDES
    shape = list(positions.shape)
    shape[1] = 0
    new_positions = np.empty( shape = shape, dtype = float )
    for x in [-1,0,1]:
        new_x = positions[...,0] + x*box[:,np.newaxis,0]
        for y in [-1,0,1]:
            new_y = positions[...,1] + y*box[:,np.newaxis,1]
            if is_2d:
                new_xy = np.stack( ( new_x, new_y ), axis=2 )
                new_positions = np.hstack(( new_positions, new_xy ))
            else:
                for z in [-1,0,1]:
                    new_z = positions[...,2] + z*box[:,np.newaxis,2]
                    new_xyz = np.stack( ( new_x, new_y, new_z ), axis=2 )
                    new_positions = np.hstack(( new_positions, new_xyz ))
                    
    return new_positions

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE CLASS
    voronoi = SamVoronoi( **kwargs )

    ## VISUALIZE VORONOI DIAGRAM
    vor, ligand_labels = voronoi.tessellation()
    voronoi.plot_diagram( vor, labels = ligand_labels )
    
    ## COMPUTE AREAS
    areas = voronoi.area( per_frame = False )

#    ## COMPUTE AREAS PER FRAME (GENERATE ~1.5 GB FILE FOR 10,OOO FRAMES)
#    areas_per_frame = voronoi.area( per_frame = True )

    ## COMPUTE SAM COMPOSITION
    composition = voronoi.polar_composition()
    
    ## RETURN RESULTS
    return { "voronoi_areas"           : areas,
#             "voronoi_areas_per_frame" : areas_per_frame,
             "sam_polar_composition"   : composition }
    
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

    ## TRAJ PREFIX
    input_prefix = "sam_prod"
    
    ## LOAD TRAJ
    traj = load_md_traj( path_traj    = path_traj,
                         input_prefix = input_prefix )

    ## INITIALIZE CLASS
    voronoi = SamVoronoi( traj              = traj,
                          sim_working_dir   = path_traj,
                          input_prefix      = input_prefix,
                          dimensions        = 2,
                          periodic          = True,
                          recompute_voronoi = True )
    
    ## VISUALIZE VORONOI DIAGRAM
    vor, ligand_labels = voronoi.tessellation()
    voronoi.plot_diagram( vor, labels = ligand_labels )
    
    ## COMPUTE AREAS
    areas = voronoi.area( per_frame = False )
    
    ## COMPUTE AREAS PER FRAME
    areas_per_frame = voronoi.area( per_frame = True )
    
    ## COMPUTE SAM COMPOSITION
    composition = voronoi.polar_composition()
    
    ## PRINT OUT RESULTS
    print( "\nAVERAGE VORONOI AREA: {}".format( areas.mean() ) )
    print( "VORORNOI AREAS PER FRAME SIZE: {}".format( areas_per_frame.shape[0] ) )
    print( "SAM POLAR COMPOSITION: {}".format( composition ) )
    