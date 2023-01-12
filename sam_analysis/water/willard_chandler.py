"""
willard_chandler.py 
script contains willard-chandler class with methods to compute the instantaneous
WC interface, averaged WC interface, and visualizing

CREATED ON: 09/25/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:
"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORTING PYTHON MODULES
import os
import numpy as np
## SCIPY FUNCTIONS
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
## MARCHING CUBES
from skimage.measure import marching_cubes_lewiner as marching_cubes

## IMPORT TRAJECTORY FUNCTION
from sam_analysis.core.trajectory import load_md_traj
### IMPORT CREATE PDB FUNCTION
#from sam_analysis.core.read_write_tools import create_pdb
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORT SERIAL AND PARALLEL FUNCTIONS
from sam_analysis.core.parallel import run_parallel
## IMPORT MISC TOOLS
from sam_analysis.core.misc_tools import track_time, fixed_grid

##############################################################################
## FUNCTIONS AND CLASSES
##############################################################################
## WILLARD-CHANDLER CLASS OBJECT
class WillardChandler:
    """class object used to compute the Willard-Chandler interface"""
    def __init__( self,
                  traj                = None,
                  sim_working_dir     = None,
                  input_prefix        = None,
                  n_procs             = 1,
                  alpha               = 0.24,
                  contour             = 16.0,
                  mesh                = [0.1, 0.1, 0.1],
                  recompute_interface = False,
                  **kwargs ):
        ## INITIALIZE VARIABLES IN CLASS
        self.traj                = traj
        self.sim_working_dir     = sim_working_dir
        self.input_prefix        = input_prefix
        self.n_procs             = n_procs
        self.alpha               = alpha
        self.contour             = contour
        self.mesh                = mesh
        self.recompute_interface = recompute_interface
        
        ## UPDATE OBJECT WITH KWARGS
        self.__dict__.update(**kwargs)

    def __str__( self ):
        return "Willard-Chandler Interface"        

    ## FUNCTION TO COMPUTE WC INTERFACE GRID
    def grid( self,
              per_frame = False,
              path_pdb  = None,
              path_cube = None, ):
        ## PATH TO PKL
        if per_frame is not True:
            out_name = self.input_prefix + "_willard_chandler_grid.pkl"
        else:
            out_name = self.input_prefix + "_willard_chandler_grid_per_frame.pkl"
        path_pkl = os.path.join( self.sim_working_dir, "output_files", out_name )
        
        ## LOAD DATA
        if self.recompute_interface is True or \
           os.path.exists( path_pkl ) is not True:
            ## LOAD TRAJ IN NOT INPUT
            if self.traj is None:
                self.traj = load_md_traj( path_traj    = self.sim_working_dir,
                                          input_prefix = self.input_prefix )
                
            ## REDUCE TRAJ TO ONLY OXYGEN ATOMS
            ## EXTRACT OXYGEN ATOM INDICES IN RESIDUE LIST
            atom_indices = np.array([ atom.index for atom in self.traj.topology.atoms 
                                       if atom.residue.name in [ "SOL", "HOH" ]
                                       and atom.element.symbol == "O" ])
            
            ## REDUCE TRAJECTORY TO ONLY ATOMS OF INVOLVED IN CALCULATION AND LAST 20% OF FRAMES
            self.traj = self.traj[-1000:].atom_slice( atom_indices, inplace = False )
                
            ## INITIALIZE WILLARD-CHANDLER OBJECT
            density_field_obj = DensityField( **self.__dict__ )
            
            ## COMPUTE DENSITY FIELDS
            if self.n_procs > 1:
                results = run_parallel( density_field_obj, self.traj, self.n_procs )
            else:
                results = density_field_obj.compute( self.traj )
                            
            if per_frame is not True:
                ## AVERAGE DENSITY FIELD
                avg_density_field = np.mean( results, axis = 0 )
                
                ## RESHAPE DENSITY FIELD
                avg_density_field = avg_density_field.reshape(density_field_obj.num_grid_pts)
                
#                ## WRITE GAUSSIAN CUBE FILE
#                if path_cube is not None:
#                    create_cube( avg_density_field, path_cube, traj )
                
                ## COMPUTE INTERFACE AT CONTOUR
                grid = self.find_contour( avg_density_field,
                                          density_field_obj.spacing )
                    
                ## UPDATE GRID (ASSUMES SINGLE SAM SYSTEM)
                grid = grid[grid[:,2] < grid[:,2].mean()]
                
                ## FIXED GRID
                grid = fixed_grid( grid, 
                                   xrange = [ 0, self.traj.unitcell_lengths[0,0] ],
                                   yrange = [ 0, self.traj.unitcell_lengths[0,1] ],
                                   spacing = 0.1 )
            else:
                ## CREATE LIST TO HOLD GRIDS
                grid = []
                ## LOOP THROUGH EACH DENSITY FIELD
                for df in results:                
                    ## RESHAPE DENSITY FIELD
                    density_field = df.reshape(density_field_obj.num_grid_pts)
                    
#                    ## WRITE GAUSSIAN CUBE FILE
#                    if path_cube is not None:
#                        create_cube( density_field, path_cube, traj )
                    
                    ## COMPUTE INTERFACE AT CONTOUR
                    gg = self.find_contour( density_field,
                                            density_field_obj.spacing )
                    
                    ## UPDATE GRID (ASSUMES SINGLE SAM SYSTEM)
                    gg = gg[gg[:,2] < gg[:,2].mean()]
                    
                    ## FIXED GRID
                    fixed_gg = fixed_grid( gg, 
                                           xrange = [ 0, self.traj.unitcell_lengths[0,0] ],
                                           yrange = [ 0, self.traj.unitcell_lengths[0,1] ],
                                           spacing = 0.1 )
                    
                    ## ADD GRID TO LIST
                    grid.append( fixed_gg )
                
                ## CONVERT TO NUMPY ARRAY
                grid = np.array( grid )
                    
            ## SAVE GRID TO PKL
            save_pkl( grid, path_pkl )
            
#            ## WRITE PDB FILE
#            if path_pdb is not None:
#                create_pdb( grid, path_pdb, traj )
        else:
             ## LOAD PKL
            grid = load_pkl( path_pkl )  
    
        ## RETURN RESULTS
        return grid

    ### FUNCTION TO COMPUTE THE WILLARD-HANDLER INTERFACE
    def find_contour( self,
                      density_field,
                      spacing       ):
        '''
        The purpose of this function is to compute the contour of the WC interface. 
        INPUTS:
            density_field: [np.array, shape = (num_grid_x, num_grid_y, num_grid_y) ]
                density values as a function of grid point
            spacing: [np.array]
                spacing in the mesh grid
            contour: [float]
                c value in the WC interface. 16 would be half the bulk. 
        OUTPUTS:
            verts: [np.array, shape = (num_atoms, 3)]
                points of the marching cubes
        '''
        ## USING MARCHING CUBES
        verts, faces, normals, values = marching_cubes( density_field, 
                                                        level   = self.contour, 
                                                        spacing = tuple( spacing ) )
        ## RETURN VERTS
        return verts
    
## DENSITY FIELD CLASS
class DensityField:
    '''
    The purpose of this class is to generate a density field by coarse-graining
    water density
    ASSUMPTIONS:
        - You have a NVT ensemble, so box does not change
        - You do not have atoms appearing and being destroyed (normal for MD simulations)
    '''
    ## INITIALIZING
    def __init__( self,
                  traj,
                  alpha      = 0.24, 
                  mesh       = [ 0.1, 0.1, 0.1 ],
                  verbose    = True, 
                  print_freq = 100,
                  **kwargs ):
        
        ## STORING INPUTS IN CLASS
        self.alpha      = alpha
        self.mesh       = mesh
        self.verbose    = verbose
        self.print_freq = print_freq       

        ## THIS ALGORITHM USES A CONSTANT BOX SIZE AND GRID SPACING, SO THIS 
        ## ONLY NEEDS TO BE GENERATED ONCE
        ## GETTING BOX LENGTHS
        self.box = traj.unitcell_lengths[ 0, : ] # ASSUME BOX DOES NOT CHANGE!
        
        ## COMPUTING MESH PARAMETERS
        self.num_grid_pts, self.spacing = compute_compatible_mesh_params( mesh = self.mesh,
                                                                          box  = self.box )
        ## CREATING GRID POINTS
        self.grid = create_grid_of_points( box = self.box,
                                           num_grid_pts = self.num_grid_pts )

    def __str__( self ):
        return "Density field"
    
    ### FUNCTION TO COMPUTE DENSITY FOR A SINGLE FRAME
    def compute_single_frame( self, traj, frame = 0 ):
        """The purpose of this function is to compute a density field for a 
        single frame"""
        ## REDUCE TRAJ TO SINGLE FRAME
        traj = traj[frame]
        
        ## GETTING DENSITIES
        density_field = compute_density_field( grid  = self.grid,
                                               pos   = traj.xyz[0,:,:],
                                               alpha = self.alpha, 
                                               box   = self.box )             
        ## RETURN RESULTS
        return density_field

    ## FUNCTION TO COMPUTE FOR ALL FRAMES
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
                density_field = [self.compute_single_frame( traj = traj, frame = frame )]
            else:
                ## COMPUTING TRIPLET ANGLES AND CONCATENATE
                density_field += [self.compute_single_frame( traj = traj, frame = frame )]
                
            ## PRINT PROGRESS IF VERBOSE
            if self.verbose is True and traj.time[frame] % self.print_freq == 0:
                 print( "  PID {}: Analyzing frame at {} ps".format( os.getpid(), traj.time[frame] ) )

        if self.verbose is True:
            ## OUTPUTTING TIME
            timer.time_elasped( "  PID " + str(os.getpid()) )
                
        ## RETURN RESULTS
        return density_field

### FUNCTION TO GENERATE DENSITY FIELD
def compute_density_field( grid, pos, alpha, box ):
    '''
    The purpose of this function is to compute the density field
    INPUTS:
        grid: [np.array, shape = (3, num_points)]
            grid points in, x,y,z dimensions.
        pos: [np.array, shape = (N_atoms, 3)]
            positions of the water
        alpha: [float]
            standard deviation of the points
    OUTPUTS:
        density_field: [np.array, shape=num_grid_points]
            density field as a function of grid points
    '''
    ## GETTING DISTRIBUTION
    dist = compute_gaussian_kde_pbc( grid  = grid,
                                     pos   = pos,
                                     alpha = alpha,
                                     box   = box )
    ## GETTING DENSITY FIELD (NORMALIZED)
    density_field = ( 2 * np.pi * alpha**2 )**(-1.5) * dist
    ## RETURN RESULTS
    return density_field
    
## GENERATING COMPATIBLE MESH PARAMETETRS
def compute_compatible_mesh_params( mesh, box ):
    """ 
    The purpose of this function is to determine the number of grid points 
    for the mesh in x, y, and z dimensions. The mesh size and box size 
    are taken into account.
    INPUTS:
        mesh: [np.array, shape = 3]
            mesh in x, y, z dimensions
        box: [np.array, shape = 3]
    OUTPUTS:
        num_grid_pts: [np.array, shape=3]
            number of grid points in x, y, z dimensions
        spacing: [np.array, shape=3]
            spacing in x, y, z dimensions
    """
    ## GETTING THE UPPER BOUND OF THE GRID POINTS
    num_grid_pts = np.ceil( box / mesh ).astype('int')
    ## GETTING SPACING BETWEEN GRID POINTS
    spacing = box / num_grid_pts
    return num_grid_pts, spacing
    
### FUNCTION TO CREATE GRID POINTS
def create_grid_of_points( box,num_grid_pts ):
    '''
    The purpose of this function is to create a grid of points given the box 
    details. 
    INPUTS:
        box: [np.array, shape = 3]
            the simulation box edges
        num_grid_pts: [np.array, shape = 3]
            number of grid points
    OUTPUTS:
        grid: [np.array, shape = (3, num_points)]
            grid points in x, y, z positions
    '''    
    ## CREATING XYZ POINTS OF THE GRID
    xyz = [ np.linspace( 0, box[each_axis], int(num_grid_pts[each_axis] ), endpoint = False )
             for each_axis in range( len(num_grid_pts) ) ]    
    '''
    NOTE1: If endpoint is false, it will not include the last value. This is 
    important for PBC concerns
    NOTE2: XYZ was generated slightly differently in Brad's code. He included a 
    subtraction of  box[i] / num_grid_pts[i] (equivalent to spacing). This is no 
    longer necessary with endpoint set to false
    '''
    ## CREATING MESHGRID OF POINTS
    x, y, z = np.meshgrid( xyz[0], xyz[1], xyz[2], indexing = "ij" ) 
    ## Indexing "xy" messes up the order!
    
    ## CONCATENATING ALL POINTS
    grid = np.concatenate(( x.reshape(1, -1),
                            y.reshape(1, -1),
                            z.reshape(1, -1)),
                            axis = 0 ) ## SHAPE: 3, NUM_POINTS
    return grid
    
### FUNCTION TO GET DENSITIES
def compute_gaussian_kde_pbc( grid,
                              pos,
                              alpha,
                              box,
                              cutoff_factor = 2.5 ):
    '''
    This function computes the Gaussian KDE values using Gaussian distributions. 
    This takes into account periodic boundary conditions
    INPUTS:
        grid: [np.array, shape = (3, num_points)]
            grid points in, x,y,z dimensions.
        pos: [np.array, shape = (N_atoms, 3)]
            positions of the water
        alpha: [float]
            standard deviation of the points
        box: [np.array, shape = 3]
            box size in x, y, z dimension
        cutoff_factor: [float]
            cutoff of standard deviations. By default, this is 2.5, which 
            is 2.5 standard deviations (~98.75% of population). Decreasing 
            this value will decrease the accuracy. Increasing this value will 
            make it difficult to compute nearest neighbors.
    OUTPUTS:
        dist: [np.array]
            normal density for the grid of points. Note that this distribution 
            is NOT normalized. It is just the exponential:
                np.exp( -sum(delta r) / scale )
            You will need to normalize this afterwards.
    '''
        
    ## GETTING KC TREE: FINDS ALL OF DISTANCE R WITHIN X -- QUICK NN LOOK UP
    tree = cKDTree( data    = grid.T, 
                    boxsize = box)
    
    ## DEFINING THE SCALE (VARIANCE)
    scale = 2. * alpha**2
    
    ## GETTING RADIUS (2.5 alpha gives you radius cutoff of 98.75%)
    d = alpha * cutoff_factor
    # 2.5 standard deviations truncation, equivalent to ~99% of the population
    # If you increased STD 
    
    ## GETTING THE INDICES LIST FOR ALL POINTS AROUND THE POSITIONS (MULTIPROCESSED)
    indlist = tree.query_ball_point( pos, r = d, n_jobs = -1 )
    
    ## DEFINING RESULTS (SHAPE = NUM_POINTS)
    dist = np.zeros( grid.shape[1], dtype = float )
    
    ## LOOPING THROUGH THE LIST
    for n, ind in enumerate( indlist ):
        ## GETTING DIFFERENCE OF R
        dr = grid.T[ind, :] - pos[n]
        ## POINTS GREATER THAN L / 2
        cond = np.where( dr > box / 2. )
        dr[cond] -= box[cond[1]]
        
        ## POINTS LESS THAN -L / 2
        cond = np.where( dr < -box / 2. )
        dr[cond] += box[cond[1]]
        
        ## DEFINING THE GAUSSIAN FUNCTION
        dens = np.exp( -np.sum( dr * dr, axis = 1 ) / scale )
        dist[ind] += dens
    
    return dist

## MAIN FUNCTION
def main(**kwargs):
    """Main function to run production analysis"""
    ## INITIALIZE WILLARD-CHANDLER OBJECT
    willard_chandler = WillardChandler( **kwargs )
    
    ## COMPUTE AVERAGED GRID
    grid = willard_chandler.grid()
    
    ## COMPUTE AVERAGED GRID PER FRAME
    grid_per_frame = willard_chandler.grid( per_frame = True )

    ## FLUCTUATION
    z_fluct = np.nanvar( grid_per_frame[...,2], axis = 0 )
    grid_fluctuation = np.hstack(( grid[:,:2], z_fluct[:,np.newaxis] ))
    
    ## RETURN RESULTS
    return { "wc_grid"           : grid,
             "wc_fluctuation"    : grid_fluctuation, 
             "wc_grid_per_frame" : grid_per_frame }

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
        
    ## RUN DENSITY FIELD OBJECT
    density_field_obj = DensityField( traj,
                                      alpha      = 0.24, 
                                      mesh       = [ 0.1, 0.1, 0.1 ],
                                      verbose    = True, 
                                      print_freq = 10, )
    
    ## DENSITY FIELD CAN ONLY HANDLE A SINGE TRAJ FRAME
    density_field = density_field_obj.compute_single_frame( traj, frame = 0 )
    
    ## RUN WILLARD-CHANDLER OBJECT IN SERIAL
    willard_chandler = WillardChandler( traj                = traj,
                                        sim_working_dir     = path_traj,
                                        input_prefix        = input_prefix,
                                        n_procs             = 1,
                                        alpha               = 0.24,
                                        contour             = 16.0,
                                        mesh                = [0.1, 0.1, 0.1],
                                        recompute_interface = True,
                                        print_freq          = 10, )
    grid_serial = willard_chandler.grid()

    ## RUN WILLARD-CHANDLER OBJECT IN PARALLEL
    willard_chandler = WillardChandler( traj                = traj,
                                        sim_working_dir     = path_traj,
                                        input_prefix        = input_prefix,
                                        n_procs             = 4,
                                        alpha               = 0.24,
                                        contour             = 16.0,
                                        mesh                = [0.1, 0.1, 0.1],
                                        recompute_interface = True,
                                        print_freq          = 10, )    
    grid_parallel = willard_chandler.grid()
    
    ## RUN WILLARD-CHANDLER OBJECT PER FRAME IN SERIAL
    willard_chandler = WillardChandler( traj                = traj,
                                        sim_working_dir     = path_traj,
                                        input_prefix        = input_prefix,
                                        n_procs             = 1,
                                        alpha               = 0.24,
                                        contour             = 16.0,
                                        mesh                = [0.1, 0.1, 0.1],
                                        recompute_interface = True,
                                        print_freq          = 10, )
    grid_per_frame = willard_chandler.grid( per_frame = True )
    
    ## CHECK SERIAL AND PARALLEL RUNS
    print( "SERIAL-PARALLEL MATCH: {}".format( str(np.all( grid_serial == grid_parallel )) ) )
    
    ## PRINT OUT RESULTS
    print( "DENSITY FIELD DIMENSIONS: {}".format( density_field.shape ) )
    print( "GRID DIMENSIONS: {}".format( grid_serial.shape ) )
    print( "WC INTERFACE PER FRAME SIZE: {}".format( grid_per_frame.shape[0] ) )
    
    ## PLOT GRID
    ## CHECK IF RUNNING FROM SPYDER
    from sam_analysis.core.check_tools import check_spyder
    
    ## PLOT IF USING SPYDER
    if check_spyder() is True:
        from matplotlib import cm
        import matplotlib.pyplot as plt
        
        colormap = cm.coolwarm
    
        xgrid = grid_serial[:,0]
        ygrid = grid_serial[:,1]
        zgrid = grid_serial[:,2]
        
        extent = [ xgrid.min(), xgrid.max(),
                   ygrid.min(), ygrid.max() ]
        
        ncols = np.unique(xgrid).shape[0]
        
        ## RESHAPE
        Z = zgrid.reshape( -1, ncols )
        
        plt.figure()
        c = plt.imshow( Z, cmap = colormap, vmin = np.nanmin(zgrid), vmax = np.nanmax(zgrid),
                        extent = extent, interpolation = "nearest", 
                        origin = "lower" )
        plt.colorbar( c )
    