# -*- coding: utf-8 -*-
"""
fnxl_grp_area.py
script to calculate area occupied by functional groups on SAMs

"""
##############################################################################
# Imports
##############################################################################
import sys, os
if "linux" in sys.platform and "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use('Agg') # turn off interactive plotting

## ADD PATH TO SYS
if r"R:/bin/python_modules" not in sys.path:
    sys.path.append( r"R:/bin/python_modules" )
    
from sam_analysis.tools.voronoi import voronoi # Loading itp reading tool
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

#%%
##############################################################################
# Load analysis inputs and trajectory
##############################################################################   
if __name__ == "__main__":
    # --- TESTING ---
    wd = r'R:\simulations\polar_sams\indus\sam_single_12x12_separated_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36_2x2x0.3nm\sample1'
#    wd = r'/mnt/r/simulations/physical_heterogeneity/autocorrelation/sam_single_8x8_300K_dodecanethiol_tip3p_nvt_CHARMM36_0.1ps_2/'
#    wd = sys.argv[1]
    gro_file = os.path.join( wd, "equil", "sam_equil_whole.gro" )
    xtc_file = os.path.join( wd, "equil", "sam_equil_whole.xtc" )
    out_path = os.path.join( wd, "output_files" )
    coords_file = os.path.join( out_path, "cavity_coordinates.csv" )
    dims_file = os.path.join( out_path, "cavity_dimensions.csv" )
    traj = md.load( xtc_file, top = gro_file )
  
##############################################################################
# Execute/test the script
##############################################################################
#%%
    ## READ IN CAVITY POSITION FILES
    with open( coords_file ) as raw_data:
        data = raw_data.readlines()
    coords = data[0].split(':')[-1]
    coords = np.array( [ float(el) for el in coords.split(',') ] )
    print( "Origin of cavity: {:.3f},{:.3f},{:.3f}".format( coords[0], coords[1], coords[2] ) )
    
    ## READ IN CAVITY DIMENSION FILES    
    with open( dims_file ) as raw_data:
        data = raw_data.readlines()
    dimensions = data[0].split(':')[-1]
    dimensions = np.array( [ float(el) for el in dimensions.split(',') ] )
    print( "Cavity dimensions: {:.3f},{:.3f},{:.3f}".format( dimensions[0], dimensions[1], dimensions[2] ) )

    ## COMPUTE THE VORONOI DIAGRAM    
#    tail_groups = [ [ "C35", "H36", "H37", "H38" ], [ "N41", "H42", "H43" ] ]
#    tail_groups = [ [ "C35", "H36", "H37", "H38" ], [ "N41", "H42", "H43", "H44" ] ]
    tail_groups = [ [ "C35", "H36", "H37", "H38" ], [ "C38", "O39", "N40", "H41", "H42" ] ]
#    tail_groups = [ [ "C35", "H36", "H37", "H38" ], [ "O41", "H42" ] ]

#%%    
    voron = voronoi( out_path, cavity_center = coords, cavity_dimensions = dimensions, plot = True )
    _, _, _, _, areas = voron.compute( traj, tail_groups = tail_groups, out_name = "voronoi_diagram" )
    
#    a = []
#    arr = np.linspace( 0, traj.unitcell_lengths.mean(axis=0)[0], 50 )
#    for x in arr:
#        coords[0] = x
#        voron = voronoi( out_path, cavity_center = coords, cavity_dimensions = dimensions, plot = False )
#        _, _, _, _, areas = voron.compute( traj, tail_groups = tail_groups, out_name = "voronoi_diagram" )
#        a.append( areas )
#    
#    a = np.array(a)
#    ratio = a[:,1] / a[:,0]
#    plt.plot( arr, ratio )

#%%    
#    target_composition = 0.75
#    min_ndx = np.argmin( np.abs(ratio[arr < 0.5*arr.max()] - target_composition) )
##    min_ndx = len(ratio[arr < 0.5*arr.max()]) + np.argmin( np.abs(ratio[arr > 0.5*arr.max()] - target_composition) )
#    print( "X position: %s" % ( str(arr[min_ndx]) ) )
#    
#    plt.plot( arr[min_ndx], ratio[min_ndx], linestyle = "None", marker = "o", color = "grey" )
    